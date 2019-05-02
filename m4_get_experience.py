import argparse
import numpy as np
import os


class m4_get_experience:
    def __init__(self, cfg):
        self.DatasetDir = cfg.DatasetDir
        self.DatasetName = cfg.DatasetName


    def m4_DirectionActionExecude(self, filename, actions_set):
        '''
        :param filename: name of the data file (one person)
        :param actions_set: action list
        :return:
        '''
        nearstate_data = np.loadtxt(filename, dtype=np.float32)
        frames, cols = nearstate_data.shape
        experience_all = np.empty(shape=[0,53],dtype=np.float32)
        for time_idx in range(frames-1):
            CurrentState = nearstate_data[time_idx]
            NextStateFact = nearstate_data[time_idx + 1]
            for action in actions_set:
                m4_CurrentActionFactR = action / 180.0 * np.pi  # 角度转弧度
                CurrentFactDirection = CurrentState[4]  # agent的当前运动方向:即从当前状态指向下一个状态的方向
                m4_DeltaAngle = np.abs(CurrentFactDirection - m4_CurrentActionFactR)  # 期望方向与动作方向的差
                m4_DeltaAngle = np.min([(2 * np.pi - m4_DeltaAngle), m4_DeltaAngle])

                if m4_DeltaAngle < 0.03490658:  # 0.01745329为1°
                    is_done = True
                    m4_NextState = NextStateFact
                    m4_Reward = 100.0
                else:
                    is_done = False
                    m4_NextState = CurrentState
                    m4_Reward = -10.0 * m4_DeltaAngle

                experience = self.m4_Store_F(CurrentState, action, m4_Reward, m4_NextState, is_done)
                experience_all = np.append(experience_all,experience,axis=0)
            print('Frame: {} done....'.format(time_idx))
        return experience_all


    def m4_Store_F(self, CurrentState, CurrentAction, Reward, NextState, done):
        if done == True:
            done = 1
        if done == False:
            done = 0
        CurrentStateList = CurrentState.tolist()  # array数组转list
        CurrentActionList = [CurrentAction]  # array数组转list
        RewardList = [Reward]
        NextStateList = NextState.tolist()  # array数组转list
        doneList = [done]
        m4_listTemp = [CurrentStateList + CurrentActionList + RewardList + NextStateList + doneList]  # list合并
        m4_listTemp_np = np.array(m4_listTemp, dtype=np.float32)
        return m4_listTemp_np

    def m4_save_direction_experience_file(self, num_person, actions_set, DatasetDir, DatasetName, save_file_path):
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        for idx in range(1, num_person+1):
            file_name = str(idx) + '.txt'
            file_name = os.path.join(DatasetDir, DatasetName, file_name)
            experience = self.m4_DirectionActionExecude(file_name, actions_set)
            np.savetxt(os.path.join(save_file_path, str(idx)+'_DirectionExperience.txt'), experience, fmt='%.6f')
            print(os.path.join(save_file_path, str(idx)+'_DirectionExperience.txt'))
        print('Done!')


    def m4_VelocityActionExecude(self, filename, actions_set):
        '''
        :param filename: name of the data file (one person)
        :param actions_set: action list
        :return:
        '''
        nearstate_data = np.loadtxt(filename, dtype=np.float32)
        frames, cols = nearstate_data.shape
        experience_all = np.empty(shape=[0,53],dtype=np.float32)
        for time_idx in range(frames-1):
            CurrentState = nearstate_data[time_idx]
            NextStateFact = nearstate_data[time_idx + 1]
            for action in actions_set:
                m4_CurrentActionFactR = action  # 角度转弧度
                CurrentFactVelocity = np.sqrt(CurrentState[2] ** 2 + CurrentState[3] ** 2) # current velocity
                m4_DeltaVelocity = np.abs(CurrentFactVelocity - m4_CurrentActionFactR)  # 期望方向与动作方向的差


                if m4_DeltaVelocity < 0.02:  #
                    is_done = True
                    m4_NextState = NextStateFact
                    m4_Reward = 100.0
                else:
                    is_done = False
                    m4_NextState = CurrentState
                    m4_Reward = -10.0 * m4_DeltaVelocity

                experience = self.m4_Store_F(CurrentState, action, m4_Reward, m4_NextState, is_done)
                experience_all = np.append(experience_all,experience,axis=0)
            print('Frame: {} done....'.format(time_idx))
        return experience_all

    def m4_save_velocity_experience_file(self, num_person, actions_set, DatasetDir, DatasetName, save_file_path):
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        for idx in range(1, num_person+1):
            file_name = str(idx) + '.txt'
            file_name = os.path.join(DatasetDir, DatasetName, file_name)
            experience = self.m4_VelocityActionExecude(file_name, actions_set)
            np.savetxt(os.path.join(save_file_path, str(idx)+'_VelocityExperience.txt'), experience, fmt='%.6f')
            print(os.path.join(save_file_path, str(idx)+'_VelocityExperience.txt'))
        print('Done!')




if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--DatasetDir',default='F:/Pedestrians_Data/Non_Reward/Convection',
                       help='near state dataset save dir')
    parse.add_argument('--DatasetName', default='Convection1/', help='near state dataset name')
    parse.add_argument('--SaveDirecitonDir', default='F:/Pedestrians_Data_Processed/Non_Reward/Convection',
                       help='experience data save dir')
    parse.add_argument('--SaveDirecitonName', default='Convection1/', help='experience data set name')

    parse.add_argument('--SaveVelocityDir', default='F:/Pedestrians_Data_Processed/Non_Reward/Convection',
                       help='experience data save dir')
    parse.add_argument('--SaveVelocityName', default='Convection1/', help='experience data set name')

    parse.add_argument('--num_DirectionAction', default=180, type=int, help='number of action')
    parse.add_argument('--num_VelocityAction', default=201, type=int, help='number of action')
    parse.add_argument('--num_person', default=26, type=int, help='number of person')

    args = parse.parse_args()

    data = m4_get_experience(args)
    m4_DirectionActionSet = np.linspace(0, 358, args.num_DirectionAction)
    m4_VelocityActionSet = np.linspace(0, 4, args.num_VelocityAction)

    data.m4_save_velocity_experience_file(args.num_person, m4_VelocityActionSet, args.DatasetDir,
                                          args.DatasetName, os.path.join(args.SaveVelocityDir,
                                                                         args.SaveVelocityName))
    data.m4_save_direction_experience_file(args.num_person, m4_DirectionActionSet, args.DatasetDir,
                                           args.DatasetName, os.path.join(args.SaveDirecitonDir,
                                                                          args.SaveDirecitonName))

