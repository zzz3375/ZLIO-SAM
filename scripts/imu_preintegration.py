#!/home/zzz2004/miniconda3/envs/pin/bin/python

import rospy
import numpy as np
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, List

import gtsam
from gtsam.symbol_shorthand import X, V, B
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped

G_MS2 = 9.81

P_WI = np.eye(4)
R_WI = P_WI[:3, :3]  # Extract rotation matrix

class ParamServer:
    """参数服务器模拟类"""
    def __init__(self):
        # 从ROS参数服务器获取参数
        self.lidar_frame = rospy.get_param('~lidar_frame', 'base_link')
        self.baselink_frame = rospy.get_param('~baselink_frame', 'base_link')
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.odometry_frame = rospy.get_param('~odometry_frame', 'odom')
        
        self.imu_topic = rospy.get_param('~imu_topic', 'livox/imu')
        self.odom_topic = rospy.get_param('~odom_topic', "odometry/imu" )
        
        # IMU参数
        self.imu_gravity = rospy.get_param('~imu_gravity', 9.81)
        self.imu_acc_noise = rospy.get_param('~imu_acc_noise', 0.01)
        self.imu_gyr_noise = rospy.get_param('~imu_gyr_noise', 0.01)
        self.imu_acc_bias_n = rospy.get_param('~imu_acc_bias_n', 0.01)
        self.imu_gyr_bias_n = rospy.get_param('~imu_gyr_bias_n', 0.01)
        
        # 外参
        ext_trans = rospy.get_param('~extrinsic_trans', [0.0, 0.0, 0.0])
        self.ext_trans = np.array(ext_trans)

@dataclass
class IMUData:
    """IMU数据结构"""
    header: rospy.Header
    linear_acceleration: Vector3
    angular_velocity: Vector3

class TransformFusion(ParamServer):
    """坐标变换融合类"""
    
    def __init__(self):
        super().__init__()
        
        self.mtx = threading.Lock()
        
        # 变换数据
        self.lidar_odom_affine = np.eye(4)
        self.imu_odom_queue: Deque[Odometry] = deque()
        self.lidar_odom_time = -1.0
        
        # TF相关
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.lidar2baselink: Optional[TransformStamped] = None
        
        # 初始化TF变换
        if self.lidar_frame != self.baselink_frame:
            try:
                self.lidar2baselink = self.tf_buffer.lookup_transform(
                    self.lidar_frame, self.baselink_frame, rospy.Time(0), rospy.Duration(3.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                   tf2_ros.ExtrapolationException) as e:
                rospy.logerr(f"TF lookup failed: {e}")
        
        # 订阅和发布
        self.sub_laser_odom = rospy.Subscriber(
            "lio_sam/mapping/odometry", Odometry, self.lidar_odometry_handler, queue_size=5)
        
        self.sub_imu_odom = rospy.Subscriber(
            f"{self.odom_topic}_incremental", Odometry, self.imu_odometry_handler, queue_size=2000)
        
        self.pub_imu_odom = rospy.Publisher(self.odom_topic, Odometry, queue_size=2000)
        self.pub_imu_path = rospy.Publisher("lio_sam/imu/path", Path, queue_size=1)
        
        # 静态TF发布
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_map2odom = tf2_ros.TransformBroadcaster()
        
        # IMU路径
        self.imu_path = Path()
        self.last_path_time = -1.0
    
    def odom_to_affine(self, odom: Odometry) -> np.ndarray:
        """将Odometry消息转换为4x4齐次变换矩阵"""
        from scipy.spatial.transform import Rotation as R
        
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        
        # 创建旋转矩阵
        rot = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        rotation_matrix = rot.as_matrix()
        
        # 创建齐次变换矩阵
        affine = np.eye(4)
        affine[:3, :3] = rotation_matrix
        affine[:3, 3] = [position.x, position.y, position.z]
        
        return affine
    
    def affine_to_odom(self, affine: np.ndarray, header: rospy.Header) -> Odometry:
        """将4x4齐次变换矩阵转换为Odometry消息"""
        from scipy.spatial.transform import Rotation as R
        
        odom = Odometry()
        odom.header = header
        
        # 提取位置
        odom.pose.pose.position.x = affine[0, 3]
        odom.pose.pose.position.y = affine[1, 3]
        odom.pose.pose.position.z = affine[2, 3]
        
        # 提取旋转并转换为四元数
        rotation = R.from_matrix(affine[:3, :3])
        quat = rotation.as_quat()
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        
        return odom
    
    def lidar_odometry_handler(self, odom_msg: Odometry):
        """激光里程计处理回调"""
        with self.mtx:
            self.lidar_odom_affine = self.odom_to_affine(odom_msg)
            self.lidar_odom_time = odom_msg.header.stamp.to_sec()
    
    def imu_odometry_handler(self, odom_msg: Odometry):
        """IMU里程计处理回调"""
        # 发布静态TF: map -> odom
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = odom_msg.header.stamp
        map_to_odom.header.frame_id = self.map_frame
        map_to_odom.child_frame_id = self.odometry_frame
        map_to_odom.transform.translation.x = 0.0
        map_to_odom.transform.translation.y = 0.0
        map_to_odom.transform.translation.z = 0.0
        map_to_odom.transform.rotation.x = 0.0
        map_to_odom.transform.rotation.y = 0.0
        map_to_odom.transform.rotation.z = 0.0
        map_to_odom.transform.rotation.w = 1.0
        
        self.tf_map2odom.sendTransform(map_to_odom)
        
        with self.mtx:
            self.imu_odom_queue.append(odom_msg)
            
            # 等待激光里程计数据
            if self.lidar_odom_time == -1:
                return
            
            # 移除过旧的IMU里程计数据
            while self.imu_odom_queue:
                if self.imu_odom_queue[0].header.stamp.to_sec() <= self.lidar_odom_time:
                    self.imu_odom_queue.popleft()
                else:
                    break
            
            if len(self.imu_odom_queue) < 2:
                return
            
            # 计算增量变换
            imu_odom_affine_front = self.odom_to_affine(self.imu_odom_queue[0])
            imu_odom_affine_back = self.odom_to_affine(self.imu_odom_queue[-1])
            imu_odom_affine_incre = np.linalg.inv(imu_odom_affine_front) @ imu_odom_affine_back
            
            # 融合激光里程计和IMU增量
            imu_odom_affine_last = self.lidar_odom_affine @ imu_odom_affine_incre
            
            # 发布融合后的里程计
            laser_odometry = self.imu_odom_queue[-1]
            fused_odom = self.affine_to_odom(imu_odom_affine_last, laser_odometry.header)
            self.pub_imu_odom.publish(fused_odom)
            
            # 发布TF: odom -> base_link
            odom_to_baselink = TransformStamped()
            odom_to_baselink.header.stamp = odom_msg.header.stamp
            odom_to_baselink.header.frame_id = self.odometry_frame
            odom_to_baselink.child_frame_id = self.baselink_frame
            odom_to_baselink.transform.translation.x = fused_odom.pose.pose.position.x
            odom_to_baselink.transform.translation.y = fused_odom.pose.pose.position.y
            odom_to_baselink.transform.translation.z = fused_odom.pose.pose.position.z
            odom_to_baselink.transform.rotation = fused_odom.pose.pose.orientation
            
            # 如果激光雷达和基座标系不同，应用变换
            if self.lidar_frame != self.baselink_frame and self.lidar2baselink:
                # 这里需要实现坐标变换组合，简化处理
                pass
                
            self.tf_broadcaster.sendTransform(odom_to_baselink)
            
            # 发布IMU路径
            imu_time = self.imu_odom_queue[-1].header.stamp.to_sec()
            if imu_time - self.last_path_time > 0.1:
                self.last_path_time = imu_time
                
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = self.imu_odom_queue[-1].header.stamp
                pose_stamped.header.frame_id = self.odometry_frame
                pose_stamped.pose = fused_odom.pose.pose
                
                self.imu_path.poses.append(pose_stamped)
                
                # 移除过旧的路径点
                while (self.imu_path.poses and 
                       self.imu_path.poses[0].header.stamp.to_sec() < self.lidar_odom_time - 1.0):
                    self.imu_path.poses.pop(0)
                
                if self.pub_imu_path.get_num_connections() > 0:
                    self.imu_path.header.stamp = self.imu_odom_queue[-1].header.stamp
                    self.imu_path.header.frame_id = self.odometry_frame
                    self.pub_imu_path.publish(self.imu_path)

class IMUPreintegration(ParamServer):
    """IMU预积分类"""
    
    def __init__(self):
        super().__init__()
        
        self.mtx = threading.Lock()
        
        # 系统状态标志
        self.system_initialized = False
        self.done_first_opt = False
        
        # 时间记录
        self.last_imu_t_imu = -1.0
        self.last_imu_t_opt = -1.0
        
        # 数据队列
        self.imu_que_opt: Deque[IMUData] = deque()
        self.imu_que_imu: Deque[IMUData] = deque()
        
        # 状态变量
        self.prev_pose: Optional[gtsam.Pose3] = None
        self.prev_vel: Optional[gtsam.Point3] = None
        self.prev_state: Optional[gtsam.NavState] = None
        self.prev_bias: Optional[gtsam.imuBias.ConstantBias] = None
        
        self.prev_state_odom: Optional[gtsam.NavState] = None
        self.prev_bias_odom: Optional[gtsam.imuBias.ConstantBias] = None
        
        # 关键帧计数
        self.key = 1
        
        # 噪声模型
        self.prior_pose_noise: Optional[gtsam.noiseModel.Diagonal] = None
        self.prior_vel_noise: Optional[gtsam.noiseModel.Isotropic] = None
        self.prior_bias_noise: Optional[gtsam.noiseModel.Isotropic] = None
        self.correction_noise: Optional[gtsam.noiseModel.Diagonal] = None
        self.correction_noise2: Optional[gtsam.noiseModel.Diagonal] = None
        self.noise_model_between_bias: Optional[np.ndarray] = None
        
        # 预积分器
        self.imu_integrator_opt: Optional[gtsam.PreintegratedImuMeasurements] = None
        self.imu_integrator_imu: Optional[gtsam.PreintegratedImuMeasurements] = None
        
        # 优化器
        self.optimizer: Optional[gtsam.ISAM2] = None
        self.graph_factors = gtsam.NonlinearFactorGraph()
        self.graph_values = gtsam.Values()
        
        # 外参变换
        self.imu2lidar = gtsam.Pose3(
            gtsam.Rot3(), 
            gtsam.Point3(-self.ext_trans[0], -self.ext_trans[1], -self.ext_trans[2]))
        self.lidar2imu = gtsam.Pose3(
            gtsam.Rot3(), 
            gtsam.Point3(self.ext_trans[0], self.ext_trans[1], self.ext_trans[2]))
        
        # 初始化
        self.initialize_models()
        self.reset_optimization()
        
        # ROS订阅和发布
        self.sub_imu = rospy.Subscriber(
            self.imu_topic, Imu, self.imu_handler, queue_size=2000)
        
        self.sub_odometry = rospy.Subscriber(
            "lio_sam/mapping/odometry_incremental", Odometry, self.odometry_handler, queue_size=5)
        
        self.pub_imu_odometry = rospy.Publisher(
            f"{self.odom_topic}_incremental", Odometry, queue_size=2000)
    
    def initialize_models(self):
        """初始化噪声模型和预积分参数"""
        # 预积分参数
        p = gtsam.PreintegrationParams.MakeSharedU(self.imu_gravity)
        p.setAccelerometerCovariance(np.eye(3) * (self.imu_acc_noise ** 2))
        p.setGyroscopeCovariance(np.eye(3) * (self.imu_gyr_noise ** 2))
        p.setIntegrationCovariance(np.eye(3) * (1e-4 ** 2))
        
        # 先验偏差
        prior_imu_bias = gtsam.imuBias.ConstantBias(np.array([0.081176, -0.0065639 , 0.98246-1])*G_MS2, np.array([0.010554, -0.010818 , 0.018882]))
        
        # 噪声模型
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]))
        self.prior_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e4)
        self.prior_bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        self.correction_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1]))
        self.correction_noise2 = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        self.noise_model_between_bias = np.array([
            self.imu_acc_bias_n* 5.0, self.imu_acc_bias_n* 5.0, self.imu_acc_bias_n* 5.0,
            self.imu_gyr_bias_n, self.imu_gyr_bias_n, self.imu_gyr_bias_n
        ])
        
        # 预积分器
        self.imu_integrator_imu = gtsam.PreintegratedImuMeasurements(p, prior_imu_bias)
        self.imu_integrator_opt = gtsam.PreintegratedImuMeasurements(p, prior_imu_bias)
    
    def reset_optimization(self):
        """重置优化器"""
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        self.optimizer = gtsam.ISAM2(parameters)
        
        self.graph_factors = gtsam.NonlinearFactorGraph()
        self.graph_values = gtsam.Values()
    
    def reset_params(self):
        """重置参数"""
        self.last_imu_t_imu = -1.0
        self.done_first_opt = False
        self.system_initialized = False
    
    def failure_detection(self, vel_cur, bias_cur: gtsam.imuBias.ConstantBias) -> bool:
        """故障检测"""
        # 确保vel_cur是numpy数组
        if hasattr(vel_cur, '__len__'):
            vel_norm = np.linalg.norm(vel_cur)
        else:
            vel_norm = abs(vel_cur)
        
        if vel_norm > 30:
            rospy.logwarn("Large velocity, reset IMU-preintegration!")
            return True
        
        acc_bias = bias_cur.accelerometer()
        gyro_bias = bias_cur.gyroscope()
        
        acc_bias_norm = np.linalg.norm(acc_bias)
        gyro_bias_norm = np.linalg.norm(gyro_bias)
        
        if acc_bias_norm > 1.0 or gyro_bias_norm > 1.0:
            rospy.logwarn("Large bias, reset IMU-preintegration!")
            return True
        
        return False
    
    def imu_converter(self, imu_raw: Imu) -> IMUData:
        """IMU数据转换器（可根据需要添加校准等处理）"""
        return IMUData(
            header=imu_raw.header,
            linear_acceleration=imu_raw.linear_acceleration,
            angular_velocity=imu_raw.angular_velocity
        )
    
    def imu_handler(self, imu_raw: Imu):
        """IMU数据处理回调"""
        with self.mtx:


            this_imu = self.imu_converter(imu_raw)
            
            self.imu_que_opt.append(this_imu)
            self.imu_que_imu.append(this_imu)
            
            if not self.done_first_opt:
                return
            
            imu_time = this_imu.header.stamp.to_sec()
            dt = (1.0 / 500.0) if self.last_imu_t_imu < 0 else (imu_time - self.last_imu_t_imu)
            self.last_imu_t_imu = imu_time
            
            # 积分单次IMU测量
            acc = np.array([
                this_imu.linear_acceleration.x,
                this_imu.linear_acceleration.y, 
                this_imu.linear_acceleration.z
            ])* G_MS2
            # acc = R_WI @ acc
            gyro = np.array([
                this_imu.angular_velocity.x,
                this_imu.angular_velocity.y,
                this_imu.angular_velocity.z
            ])
            # gyro = R_WI @ gyro

            self.imu_integrator_imu.integrateMeasurement(acc, gyro, dt)
            
            # 预测里程计
            current_state: gtsam.NavState = self.imu_integrator_imu.predict(self.prev_state_odom, self.prev_bias_odom)
            
            # 发布里程计
            odometry = Odometry()
            odometry.header.stamp = this_imu.header.stamp
            odometry.header.frame_id = self.odometry_frame
            odometry.child_frame_id = "odom_imu"
            
            # 变换IMU位姿到激光雷达坐标系
            imu_pose = current_state.pose()#gtsam.Pose3(current_state.quaternion(), current_state.position())
            lidar_pose = imu_pose.compose(self.imu2lidar)
            
            # 设置位姿
            odometry.pose.pose.position.x = lidar_pose.x()
            odometry.pose.pose.position.y = lidar_pose.y()
            odometry.pose.pose.position.z = lidar_pose.z()
            
            quat = lidar_pose.rotation().toQuaternion()
            odometry.pose.pose.orientation.x = quat.x()
            odometry.pose.pose.orientation.y = quat.y()
            odometry.pose.pose.orientation.z = quat.z()
            odometry.pose.pose.orientation.w = quat.w()
            
            # 设置速度
            odometry.twist.twist.linear.x = current_state.velocity()[0]
            odometry.twist.twist.linear.y = current_state.velocity()[1]
            odometry.twist.twist.linear.z = current_state.velocity()[2]
            
            # 设置角速度（包含偏差补偿）
            odometry.twist.twist.angular.x = (this_imu.angular_velocity.x + 
                                            self.prev_bias_odom.gyroscope()[0])
            odometry.twist.twist.angular.y = (this_imu.angular_velocity.y + 
                                            self.prev_bias_odom.gyroscope()[1])
            odometry.twist.twist.angular.z = (this_imu.angular_velocity.z + 
                                            self.prev_bias_odom.gyroscope()[2])
            
            self.pub_imu_odometry.publish(odometry)
    
    def odometry_handler(self, odom_msg: Odometry):
        """里程计处理回调"""
        with self.mtx:
            current_correction_time = odom_msg.header.stamp.to_sec()
            
            # 确保有IMU数据可以积分
            if not self.imu_que_opt:
                return
            
            # 解析激光里程计位姿
            pose = odom_msg.pose.pose
            lidar_pose = gtsam.Pose3(
                gtsam.Rot3.Quaternion(pose.orientation.w, pose.orientation.x, 
                                    pose.orientation.y, pose.orientation.z),
                gtsam.Point3(pose.position.x, pose.position.y, pose.position.z)
            )
            
            degenerate = bool(odom_msg.pose.covariance[0] == 1)
            
            # 0. 系统初始化
            if not self.system_initialized:
                self.reset_optimization()
                
                # 移除旧的IMU数据
                while self.imu_que_opt:
                    if self.imu_que_opt[0].header.stamp.to_sec() < current_correction_time:
                        self.last_imu_t_opt = self.imu_que_opt[0].header.stamp.to_sec()
                        self.imu_que_opt.popleft()
                    else:
                        break
                
                # 初始位姿
                self.prev_pose = lidar_pose.compose(self.lidar2imu)
                prior_pose = gtsam.PriorFactorPose3(X(0), self.prev_pose, self.prior_pose_noise)
                self.graph_factors.add(prior_pose)
                
                # 初始速度 - 使用Vector而不是Point3
                self.prev_vel = np.array([0.0, 0.0, 0.0])
                prior_vel = gtsam.PriorFactorVector(V(0), self.prev_vel, self.prior_vel_noise)
                self.graph_factors.add(prior_vel)
                
                # 初始偏差
                self.prev_bias = gtsam.imuBias.ConstantBias()
                prior_bias = gtsam.PriorFactorConstantBias(B(0), self.prev_bias, self.prior_bias_noise)
                self.graph_factors.add(prior_bias)
                
                # 添加初始值
                self.graph_values.insert(X(0), self.prev_pose)
                self.graph_values.insert(V(0), self.prev_vel)
                self.graph_values.insert(B(0), self.prev_bias)
                
                
                # 优化一次
                self.optimizer.update(self.graph_factors, self.graph_values)
                self.graph_factors = gtsam.NonlinearFactorGraph()
                self.graph_values = gtsam.Values()
                
                # 重置预积分器
                self.imu_integrator_imu.resetIntegrationAndSetBias(self.prev_bias)
                self.imu_integrator_opt.resetIntegrationAndSetBias(self.prev_bias)
                
                # 创建初始NavState - 使用Vector
                self.prev_state = gtsam.NavState(self.prev_pose, gtsam.Point3(0, 0, 0))
                self.prev_state_odom = self.prev_state
                self.prev_bias_odom = self.prev_bias
                
                self.key = 1
                self.system_initialized = True
                return
            
            
            # 1. 积分IMU数据并优化
            while self.imu_que_opt:
                this_imu = self.imu_que_opt[0]
                imu_time = this_imu.header.stamp.to_sec()
                
                if imu_time < current_correction_time:
                    dt = (1.0 / 500.0) if self.last_imu_t_opt < 0 else (imu_time - self.last_imu_t_opt)
                    
                    acc = np.array([
                        this_imu.linear_acceleration.x,
                        this_imu.linear_acceleration.y,
                        this_imu.linear_acceleration.z
                    ])* G_MS2
                    # acc = R_WI @ acc
                    gyro = np.array([
                        this_imu.angular_velocity.x,
                        this_imu.angular_velocity.y,
                        this_imu.angular_velocity.z
                    ])
                    # gyro = R_WI @ gyro
                    self.imu_integrator_opt.integrateMeasurement(acc, gyro, dt)
                    self.last_imu_t_opt = imu_time
                    self.imu_que_opt.popleft()
                else:
                    break
            
            # 添加IMU因子到图中
            imu_factor = gtsam.ImuFactor(
                X(self.key-1), V(self.key-1), X(self.key), V(self.key), 
                B(self.key-1), self.imu_integrator_opt)
            self.graph_factors.add(imu_factor)
            
            # 添加IMU偏差因子
            bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.sqrt(self.imu_integrator_opt.deltaTij()) * self.noise_model_between_bias)
            bias_factor = gtsam.BetweenFactorConstantBias(
                B(self.key-1), B(self.key), gtsam.imuBias.ConstantBias(), bias_noise)
            self.graph_factors.add(bias_factor)
            
            # 添加位姿因子
            cur_pose = lidar_pose.compose(self.lidar2imu)
            pose_noise = self.correction_noise2 if degenerate else self.correction_noise
            pose_factor = gtsam.PriorFactorPose3(X(self.key), cur_pose, pose_noise)
            self.graph_factors.add(pose_factor)
            
            # 插入预测值 - 修复：统一使用Vector类型
            prop_state = self.imu_integrator_opt.predict(self.prev_state, self.prev_bias)
            
            # 确保速度是Vector类型
            velocity_vector = np.array(prop_state.velocity()).T
            
            # 使用 update 方法
            if self.graph_values.exists(X(self.key)):
                self.graph_values.update(X(self.key), prop_state.pose())
            else:
                self.graph_values.insert(X(self.key), prop_state.pose())
            
            if self.graph_values.exists(V(self.key)):
                self.graph_values.update(V(self.key), velocity_vector)
            else:
                self.graph_values.insert_point3(V(self.key), velocity_vector)

            if self.graph_values.exists(B(self.key)):
                self.graph_values.update(B(self.key), self.prev_bias)
            else:
                self.graph_values.insert(B(self.key), self.prev_bias)
            
            
            # 优化
            for key in list(self.graph_values.keys()):
                if self.optimizer.valueExists(key):
                    self.graph_values.erase(key)
            self.optimizer.update(self.graph_factors, self.graph_values)
            self.optimizer.update()
            self.graph_factors = gtsam.NonlinearFactorGraph()
            self.graph_values = gtsam.Values()
            
            # 获取优化结果
            result = self.optimizer.calculateEstimate()
            self.prev_pose = result.atPose3(X(self.key))
            self.prev_vel = result.atPoint3(V(self.key))  # 直接使用Vector
            self.prev_bias = result.atConstantBias(B(self.key))
            
            # 更新NavState - 需要将Vector转换为Point3
            vel_point3 = gtsam.Point3(self.prev_vel[0], self.prev_vel[1], self.prev_vel[2])
            self.prev_state = gtsam.NavState(self.prev_pose, vel_point3)
            
            # 重置优化预积分器
            self.imu_integrator_opt.resetIntegrationAndSetBias(self.prev_bias)
            
            # 故障检测 - 需要将Vector转换为numpy数组
            vel_np = np.array([self.prev_vel[0], self.prev_vel[1], self.prev_vel[2]])
            if self.failure_detection(vel_np, self.prev_bias):
                self.reset_params()
                return
            
            # 2. 优化后重新传播IMU里程计预积分
            self.prev_state_odom = self.prev_state
            self.prev_bias_odom = self.prev_bias
            
            # 移除比当前校正数据旧的IMU消息
            last_imu_qt = -1.0
            while (self.imu_que_imu and 
                self.imu_que_imu[0].header.stamp.to_sec() < current_correction_time):
                last_imu_qt = self.imu_que_imu[0].header.stamp.to_sec()
                self.imu_que_imu.popleft()
            
            # 重新传播
            if self.imu_que_imu:
                # 使用新优化的偏差重置
                self.imu_integrator_imu.resetIntegrationAndSetBias(self.prev_bias_odom)
                
                # 从本次优化的开始积分IMU消息
                for i in range(len(self.imu_que_imu)):
                    this_imu = self.imu_que_imu[i]
                    imu_time = this_imu.header.stamp.to_sec()
                    dt = (1.0 / 500.0) if last_imu_qt < 0 else (imu_time - last_imu_qt)
                    
                    acc = np.array([
                        this_imu.linear_acceleration.x,
                        this_imu.linear_acceleration.y,
                        this_imu.linear_acceleration.z
                    ])* G_MS2
                    # acc = R_WI @ acc
                    gyro = np.array([
                        this_imu.angular_velocity.x,
                        this_imu.angular_velocity.y,
                        this_imu.angular_velocity.z
                    ])
                    # gyro = R_WI @ gyro
                    self.imu_integrator_imu.integrateMeasurement(acc, gyro, dt)
                    last_imu_qt = imu_time
            
            self.key += 1
            self.done_first_opt = True

def main():
    rospy.init_node('lio_sam_imu_preintegration', anonymous=True)
    
    # 创建IMU预积分和变换融合实例
    imu_preintegration = IMUPreintegration()
    transform_fusion = TransformFusion()
    
    rospy.loginfo("\033[1;32m----> IMU Preintegration Started.\033[0m")
    
    # 使用多线程spinner
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass