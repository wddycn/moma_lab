# 摆动脚相对身体高度约束
- **self.rewards.feet_height_body.weight = -3.0     # 原先是2.5**
- **self.rewards.feet_height_body.params["target_height"] = -0.27   # 原先是-0.33，贴近地面**
```
################################################################################
                     Learning iteration 1907/20000                      

                       Computation: 44920 steps/s (collection: 2.042s, learning 0.147s)
             Mean action noise std: 0.62
          Mean value_function loss: 0.1451
               Mean surrogate loss: -0.0055
              Mean estimation loss: 0.0367
                    Mean swap loss: 0.1019
                       Mean reward: 79.81
               Mean episode length: 1000.00
       Episode_Reward/lin_vel_z_l2: -0.0530
      Episode_Reward/ang_vel_xy_l2: -0.0879
     Episode_Reward/base_height_l2: 0.0000
   Episode_Reward/joint_torques_l2: -0.0160
       Episode_Reward/joint_acc_l2: -0.1535
   Episode_Reward/joint_pos_limits: -0.0000
        Episode_Reward/joint_power: -0.0028
        Episode_Reward/stand_still: -0.0520
  Episode_Reward/joint_pos_penalty: -0.2223
       Episode_Reward/joint_mirror: -0.0054
     Episode_Reward/action_rate_l2: -0.1559
 Episode_Reward/undesired_contacts: -0.1161
     Episode_Reward/contact_forces: -0.3329
Episode_Reward/track_lin_vel_xy_exp: 2.0235
Episode_Reward/track_ang_vel_z_exp: 1.1638
      Episode_Reward/feet_air_time: -0.1102
Episode_Reward/feet_air_time_variance: -0.0657
          Episode_Reward/feet_gait: 0.4292
Episode_Reward/feet_contact_without_cmd: 0.0012
         Episode_Reward/feet_slide: -0.1235
   Episode_Reward/feet_height_body: -0.0301
             Episode_Reward/upward: 1.9248
Episode_Reward/feet_air_without_cmd: -0.0404
         Curriculum/terrain_levels: 0.6760
Metrics/base_velocity/error_vel_xy: 0.6693
Metrics/base_velocity/error_vel_yaw: 0.4642
      Episode_Termination/time_out: 1.0000
Episode_Termination/terrain_out_of_bounds: 0.0000
--------------------------------------------------------------------------------
                   Total timesteps: 10616832
                    Iteration time: 2.19s
                      Time elapsed: 00:04:01
                               ETA: 11:13:21
```
- **self.rewards.feet_height_body.weight = -5.0     # 原先是2.5**
- **self.rewards.feet_height_body.params["target_height"] = -0.20   # 原先是-0.33，贴近地面**
```
################################################################################
                     Learning iteration 1945/20000                      

                       Computation: 43421 steps/s (collection: 2.123s, learning 0.141s)
             Mean action noise std: 0.61
          Mean value_function loss: 0.1400
               Mean surrogate loss: -0.0072
              Mean estimation loss: 0.0364
                    Mean swap loss: 0.1021
                       Mean reward: 78.05
               Mean episode length: 1000.00
       Episode_Reward/lin_vel_z_l2: -0.0611
      Episode_Reward/ang_vel_xy_l2: -0.0906
     Episode_Reward/base_height_l2: -0.0008
   Episode_Reward/joint_torques_l2: -0.0167
       Episode_Reward/joint_acc_l2: -0.1555
   Episode_Reward/joint_pos_limits: -0.0000
        Episode_Reward/joint_power: -0.0029
        Episode_Reward/stand_still: -0.0564
  Episode_Reward/joint_pos_penalty: -0.2534
       Episode_Reward/joint_mirror: -0.0056
     Episode_Reward/action_rate_l2: -0.1567
 Episode_Reward/undesired_contacts: -0.1212
     Episode_Reward/contact_forces: -0.3073
Episode_Reward/track_lin_vel_xy_exp: 2.0681
Episode_Reward/track_ang_vel_z_exp: 1.1466
      Episode_Reward/feet_air_time: -0.1048
Episode_Reward/feet_air_time_variance: -0.0659
          Episode_Reward/feet_gait: 0.4343
Episode_Reward/feet_contact_without_cmd: 0.0012
         Episode_Reward/feet_slide: -0.1242
   Episode_Reward/feet_height_body: -0.1449
             Episode_Reward/upward: 1.9301
Episode_Reward/feet_air_without_cmd: -0.0434
         Curriculum/terrain_levels: 0.3841
Metrics/base_velocity/error_vel_xy: 0.6417
Metrics/base_velocity/error_vel_yaw: 0.4869
      Episode_Termination/time_out: 1.0000
Episode_Termination/terrain_out_of_bounds: 0.0000
--------------------------------------------------------------------------------
                   Total timesteps: 4521984
                    Iteration time: 2.26s
                      Time elapsed: 00:01:44
                               ETA: 11:21:45
```