import math

def reward_function(params):
    
    if not params['all_wheels_on_track']:
        return 1e-3
        
        
    # Initialise the weights and reward 
    speed_weight = 100
    alignment_weight = 100
    centralised_weight = 75
    steering_weight = 50

    speed_reward = 0
    alignment_reward = 0
    centralised_reward = 0
    steering_reward = 0
    
    # Account for speed 
    max_speed = 4.0 * 4.0
    min_speed = 1.67 * 1.67
    abs_speed = params['speed'] * params['speed']
    
    speed_reward = speed_weight * (abs_speed - min_speed)/(max_speed - min_speed)
    
    # Account for direction 
    next_point = params['waypoints'][params['closest_waypoints'][1]]
    prev_point = params['waypoints'][params['closest_waypoints'][0]]
    
    direction = math.degrees(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))
    direction_diff = abs(direction - params['heading'])
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    alignment_reward = alignment_weight * (1 - direction_diff/180.0) 
    
    # Steering weights
    abs_steering_reward = 1 - (abs(params['steering_angle'] - direction_diff) / 180.0)
    steering_reward = abs_steering_reward * steering_weight
    
    # Stay on left but not off shortcuts
    if params['distance_from_center'] < 0.2 * params['track_width']:
        centralised_reward = 1.0
    elif params['distance_from_center'] < 0.4 * params['track_width']:
        centralised_reward = 0.5
    # Clinging to the left might be a way to run less distance
    if params['is_left_of_center']:
        centralised_reward *= 1.1
    centralised_reward = centralised_reward * centralised_weight
    
    
    
    return float(centralised_weight + speed_weight + alignment_weight + steering_reward)
