#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import time
import rospy
import atexit
from std_msgs.msg import String
real_moving_mode=None
history_modes_real=[]
history_modes_est=[]
drift_history=[]
real_x_history=[]
estimated_x_history=[]

def mode_callback(data):
   global real_moving_mode


   """
   This function is called every time a new message is published to the topic.
   """
   # The 'data' object contains the message. For a String, the value is in data.data
   #rospy.loginfo(f"Current box mode: {data.data}")
   real_moving_mode=data.data


# ------------------ IMM Model Setup ------------------
dt = 0.1


def make_cv_model():
   # State: [x, v]
   F = np.array([[1, dt], [0, 1]])
   B = np.array([[dt], [0]])  # Control input matrix for velocity
   Q = np.diag([0.001, 0.001])
   H = np.array([[1, 0]])
   R = np.array([[0.1]])
   x0 = np.zeros((2, 1))
   P0 = np.eye(2)
   return {"F": F, "B": B, "Q": Q, "H": H, "R": R, "x": x0, "P": P0}


def make_ca_model():
   # State: [x, v, a]
   F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
   B = np.array([[0.5*dt**2], [dt], [0]])  # Control input matrix
   Q = np.diag([0.0005, 0.0005,0.0005])
   H = np.array([[1, 0, 0]])
   R = np.array([[0.1]])
   x0 = np.zeros((3, 1))
   P0 = np.eye(3)
   return {"F": F, "B": B, "Q": Q, "H": H, "R": R, "x": x0, "P": P0}


def make_sg_model():
   # State: [x]
   F = np.array([[1]])
   B = np.array([[dt]])
   Q = np.array([[0.05]])
   H = np.array([[1]])
   R = np.array([[0.1]])
   x0 = np.zeros((1, 1))
   P0 = np.eye(1)
   return {"F": F, "B": B, "Q": Q, "H": H, "R": R, "x": x0, "P": P0}


def make_sd_model():
   # State: [x, v]
   F = np.array([[1, dt], [0, 1]])
   B = np.array([[dt], [0]])
   Q = np.diag([0.2, 0.2])
   H = np.array([[1, 0]])
   R = np.array([[0.1]])
   x0 = np.zeros((2, 1))
   P0 = np.eye(2)
   return {"F": F, "B": B, "Q": Q, "H": H, "R": R, "x": x0, "P": P0}


def make_zz_model():
   # State: [x, v]
   F = np.array([[1, dt], [0, 1]])
   B = np.array([[dt], [0]])
   Q = np.diag([0.5, 0.5])
   H = np.array([[1, 0]])
   R = np.array([[0.1]])
   x0 = np.zeros((2, 1))
   P0 = np.eye(2)
   return {"F": F, "B": B, "Q": Q, "H": H, "R": R, "x": x0, "P": P0}


def make_rev_model():
   # State: [x, v]
   F = np.array([[1, dt], [0, 1]])
   B = np.array([[dt], [0]])
   Q = np.diag([0.1, 0.1])
   H = np.array([[1, 0]])
   R = np.array([[0.1]])
   x0 = np.array([[0.0], [-0.01]])
   P0 = np.eye(2)
   return {"F": F, "B": B, "Q": Q, "H": H, "R": R, "x": x0, "P": P0}


def make_pt_model():
   # State: [x]
   F = np.array([[1]])
   B = np.array([[dt]])
   Q = np.array([[0.01]])
   H = np.array([[1]])
   R = np.array([[0.1]])
   x0 = np.zeros((1, 1))
   P0 = np.eye(1)
   return {"F": F, "B": B, "Q": Q, "H": H, "R": R, "x": x0, "P": P0}


model_names = ["CV", "CA", "SG", "SD", "ZZ", "REV", "PT"]
models = [
   make_cv_model(), make_ca_model(), make_sg_model(), make_sd_model(),
   make_zz_model(), make_rev_model(), make_pt_model()
]
model_probs = np.ones(len(models)) / len(models)
transition_matrix = np.ones((len(models), len(models))) * 0.02
np.fill_diagonal(transition_matrix, 0.6)
pt_index = model_names.index("PT")
transition_matrix[pt_index, pt_index] = 0.3
transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)


# ------------------ Globals ------------------
safe_distance = 0.2
resume_distance = 0.3
robot_speed = 0.07
robot_stopped = False
velocity_publisher = None
current_robot_vel_cmd = 0.0
robot_position = {"x": 0.0, "y": 0.0}
real_position = {"x": 0.0, "y": 0.0}
t = 0


# ------------------ IMM Functions ------------------
def imm_predict(model, u):
   F, B, Q = model["F"], model["B"], model["Q"]
   x, P = model["x"], model["P"]
  
   # Prediction: x_pred = Fx + Bu
   x_pred = F @ x + B @ u
   P_pred = F @ P @ F.T + Q
  
   model["x"] = x_pred
   model["P"] = P_pred
   return model


def imm_update(model, z):
   H, R, x, P = model["H"], model["R"], model["x"], model["P"]
   y = np.array([[z]]) - H @ x
   S = H @ P @ H.T + R
   K = P @ H.T @ np.linalg.inv(S)
   x_upd = x + K @ y
   P_upd = (np.eye(P.shape[0]) - K @ H) @ P
   likelihood = np.exp(-0.5 * (y.T @ np.linalg.inv(S) @ y)) / np.sqrt(np.linalg.det(2 * np.pi * S))
   model["x"] = x_upd
   model["P"] = P_upd
   return likelihood.item()


def imm_step(models, model_probs, z, u):
   M = len(models)
  
   # 1. Normalization and Mixing
   c_j = transition_matrix.T @ model_probs
   if np.any(c_j == 0):
       c_j = c_j + 1e-15
   mu_ij = (transition_matrix.T * model_probs).T / c_j
  
   mixed_models = [None] * M
   for j in range(M):
       mixed_x = np.zeros(models[j]['x'].shape)
       mixed_P = np.zeros(models[j]['P'].shape)
      
       for i in range(M):
           # Pad states and covariances to a consistent size for mixing
           # This is a key part to handle models with different dimensions
           x_i_padded = np.zeros(models[j]['x'].shape)
           x_i_padded[:models[i]['x'].shape[0], :] = models[i]['x']
           mixed_x += x_i_padded * mu_ij[i, j]
              
       for i in range(M):
           P_i_padded = np.zeros(models[j]['P'].shape)
           P_i_padded[:models[i]['P'].shape[0], :models[i]['P'].shape[0]] = models[i]['P']
           dx = x_i_padded - mixed_x
           mixed_P += mu_ij[i, j] * (P_i_padded + dx @ dx.T)
      
       mixed_models[j] = models[j].copy()
       mixed_models[j]['x'] = mixed_x
       mixed_models[j]['P'] = mixed_P


   # 2. Prediction and Update for each mixed model
   likelihoods = np.zeros(M)
   for i in range(M):
       mixed_models[i] = imm_predict(mixed_models[i], u)
       likelihoods[i] = imm_update(mixed_models[i], z)
      
   # 3. Mode Probability Update
   c = np.sum(likelihoods * c_j)
   model_probs_upd = (likelihoods * c_j) / c
  
   for i in range(M):
       models[i]['x'] = mixed_models[i]['x'][:models[i]['x'].shape[0], :]
       models[i]['P'] = mixed_models[i]['P'][:models[i]['P'].shape[0], :models[i]['P'].shape[0]]
  
   model_probs[:] = model_probs_upd
  
   return models, model_probs
def imm_step(models, model_probs, z, u):
   M = len(models)
  
   # Get the max state dimension for consistent padding
   max_dim = max([m['x'].shape[0] for m in models])


   # 1. Normalization and Mixing
   c_j = transition_matrix.T @ model_probs
   if np.any(c_j == 0):
       c_j = c_j + 1e-15
   mu_ij = (transition_matrix.T * model_probs).T / c_j
  
   mixed_models = [None] * M
   for j in range(M):
       mixed_x = np.zeros((max_dim, 1))
      
       # Weighted sum of padded states
       for i in range(M):
           x_i_padded = np.zeros((max_dim, 1))
           x_i_padded[:models[i]['x'].shape[0], :] = models[i]['x']
           mixed_x += x_i_padded * mu_ij[i, j]
              
       # Weighted sum of padded covariances
       mixed_P = np.zeros((max_dim, max_dim))
       for i in range(M):
           P_i_padded = np.zeros((max_dim, max_dim))
           P_i_padded[:models[i]['P'].shape[0], :models[i]['P'].shape[0]] = models[i]['P']
           dx = (x_i_padded - mixed_x)
           mixed_P += mu_ij[i, j] * (P_i_padded + dx @ dx.T)
      
       # Assign the mixed states to the new models
       mixed_models[j] = models[j].copy()
       mixed_models[j]['x'] = mixed_x[:models[j]['x'].shape[0], :]
       mixed_models[j]['P'] = mixed_P[:models[j]['P'].shape[0], :models[j]['P'].shape[0]]


   # 2. Prediction and Update for each mixed model
   likelihoods = np.zeros(M)
   for i in range(M):
       mixed_models[i] = imm_predict(mixed_models[i], u)
       likelihoods[i] = imm_update(mixed_models[i], z)
      
   # 3. Mode Probability Update
   c = np.sum(likelihoods * c_j)
   model_probs_upd = (likelihoods * c_j) / c
  
   # Copy the updated states back to the original models
   for i in range(M):
       models[i]['x'] = mixed_models[i]['x']
       models[i]['P'] = mixed_models[i]['P']
  
   model_probs[:] = model_probs_upd
  
   return models, model_probs






def fused_state(models, model_probs):
   fused_x, fused_v = 0.0, 0.0
   for i, m in enumerate(models):
       fused_x += model_probs[i] * m["x"][0, 0]
       if m["x"].shape[0] > 1:
           fused_v += model_probs[i] * m["x"][1, 0]
   return fused_x, fused_v


def estimated_acceleration(models, model_probs):
   acc = 0.0
   for i, m in enumerate(models):
       if m["x"].shape[0] == 3:
           acc += model_probs[i] * m["x"][2, 0]
   return acc

prev_pose_x=None
prev_pose_y=None
t=None
def odom_callback(data):
   global robot_position, current_robot_vel_cmd,prev_pose_x,prev_pose_y,t
   if t is None:
       t=time.time() 
       
   else:
   # Odom provides the robot's state, but we only use its position for global logging.
    # The velocity is now a control input to the filter.
    current_robot_vel_cmd = data.twist.twist.linear.x+np.random.normal(0.0008,0.000015)
    # robot_position["x"] = data.pose.pose.position.x
    # robot_position["y"] = data.pose.pose.position.y
    robot_position["x"] +=current_robot_vel_cmd*(time.time()-t)
    robot_position["y"] +=data.twist.twist.linear.y*(time.time()-t)
    t=time.time()
  


def model_states_callback(data):
   global real_position
   try:
       idx = data.name.index("moving_box")
       pose = data.pose[idx]
       real_position["x"] = pose.position.x
       real_position["y"] = pose.position.y
   except ValueError:
       rospy.logwarn("Box model not found in /gazebo/model_states")


def lidar_callback(data):
   global models, model_probs, robot_stopped, current_robot_vel_cmd, robot_position,real_moving_mode
   global history_modes_est,history_modes_real,drift_history,real_x_history,estimated_x_history
   # Correctly model the LiDAR measurement.
   # z = (Object's True Position) - (Robot's True Position)
   # The filter estimates the object's relative position, so z should be relative too.
   obs_rel_x = data.range

  
   # Get the control input u, which is the robot's velocity
   u = np.array([[current_robot_vel_cmd]])
  
   # Pass the relative measurement and control input to the IMM step
   models, model_probs = imm_step(models, model_probs, obs_rel_x, u)
  
   est_rel_x, est_v = fused_state(models, model_probs)
   est_a = estimated_acceleration(models, model_probs)
  
   most_likely_idx = np.argmax(model_probs)
   if model_names[most_likely_idx] == "CA":
       est_v += est_a * dt * 0.5
      
   # The IMM estimates the object's RELATIVE position to the robot.
   # To get the global position, we add the robot's global position.
   est_global_x = robot_position["x"] + est_rel_x
   model_positions = [m["x"][0, 0] for m in models]
   global_positions = [robot_position["x"] + p for p in model_positions]
  
   # Control logic
   vel_msg = Twist()
   if obs_rel_x <= safe_distance:
       vel_msg.linear.x = 0.0
       robot_stopped = True
   elif robot_stopped and obs_rel_x >= resume_distance:
       vel_msg.linear.x = robot_speed
       robot_stopped = False
   elif not robot_stopped:
       vel_msg.linear.x = robot_speed
      
   velocity_publisher.publish(vel_msg)


   # Logging
   drift_error = est_global_x - real_position["x"]
   drift_history.append(drift_error)
   real_x_history.append(real_position['x'])
   estimated_x_history.append(global_positions.append(est_global_x))
   history_modes_est.append(model_names[most_likely_idx])
   history_modes_real.append(real_moving_mode)
   '''
   rospy.loginfo(f"\n[IMM Tracker]")
   rospy.loginfo(f"  Real Position        : {real_position['x']:.3f}")
   rospy.loginfo(f"  Fused Estimate       : x={est_global_x:.3f}, v={est_v:.3f}")
   rospy.loginfo(f"  LiDAR Measurement    : {obs_rel_x:.3f}")
   rospy.loginfo(f"  Drift Error          : {drift_error:.3f}")
   '''
   rospy.loginfo(f"  Most Likely Mode     : {model_names[most_likely_idx]} - Real Mode {real_moving_mode}")
   '''
   for i, name in enumerate(model_names):
       rospy.loginfo(f"    {name:<4} → x={global_positions[i]:.3f}, prob={model_probs[i]:.2f}")
'''
import pickle
import time
def save_data():
    global history_modes_est,history_modes_real,drift_history,real_x_history,estimated_x_history
    with open(f'est_mode_{time.time()}.pkl','wb') as f:
        pickle.dump(history_modes_est,f)
    with open(f'history_modes_real_{time.time()}.pkl','wb') as f:
        pickle.dump(history_modes_real,f)
    with open(f'drift_history_{time.time()}.pkl','wb') as f:
        pickle.dump(drift_history,f)
    with open(f'real_x_history_{time.time()}.pkl','wb') as f:
        pickle.dump(real_x_history,f)
    with open(f'estimated_x_history_{time.time()}.pkl','wb') as f:
        pickle.dump(estimated_x_history,f)

if __name__ == '__main__':
   rospy.init_node('space_object_tracker')
  
   velocity_publisher = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=10)
   rospy.Subscriber('/vector/laser', Range, lidar_callback)
   rospy.Subscriber('/odom', Odometry, odom_callback)
   rospy.Subscriber('/gazebo/model_states', ModelStates, model_states_callback)


   rospy.wait_for_service("/gazebo/set_model_state")
   try:
       set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
       state_msg = ModelState()
       state_msg.model_name = 'vector'
       state_msg.pose.position.x = 0.0
       state_msg.pose.position.y = 0.0
       state_msg.pose.position.z = 0.0
       state_msg.pose.orientation.x = 0.0
       state_msg.pose.orientation.y = 0.0
       state_msg.pose.orientation.z = 0.0
       state_msg.pose.orientation.w = 1.0
       state_msg.twist.linear.x = 0.0
       state_msg.twist.linear.y = 0.0
       state_msg.twist.linear.z = 0.0
       set_state(state_msg)
       rospy.loginfo("Robot world pose set to (0,0)")
       rospy.sleep(0.5)
   except rospy.ServiceException as e:
       rospy.logerr("set_model_state failed: %s", e)
  
   rospy.loginfo("🚀 IMM Tracker Node Started. Tracking object in global frame...")
   rospy.Subscriber("box_mode", String, mode_callback)

   atexit.register(save_data)
   rospy.spin()














