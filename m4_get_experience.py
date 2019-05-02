import argparse
import numpy as np
import os


# m4_TrainSetsPath = '/media/yang/F/ubuntu/DRL_Trajectory_Prediction/v13/Near_State/TrainData.txt'
# m4_ExplorePath = '/media/yang/F/ubuntu/DRL_Trajectory_Prediction/v13/ExploreData/' + 'explore'+'.txt'
# m4_OriginalData = np.loadtxt(m4_TrainSetsPath)  # 导入原始数据
# m4_ActionTotalNumber = 361
# m4_ActionMatrix = np.identity(m4_ActionTotalNumber)  # 动作矩阵
# m4_ActionList = np.linspace(0, 360, m4_ActionTotalNumber)  # 生成动作列表
#
# m4_ExpC = m4_Explore_File.m4_ActionExecude_C()
# m4_Save = m4_Explore_File.m4_SaveStore_C()
#
# m4_Length = m4_OriginalData.shape[0] - 3
# m4_Store_np = np.zeros(shape=(1, 413))
# m4_List = []
# while True:
#     for i in range(361):
#         m4_Current, m4_Action, m4_Reward, m4_NextState, done = m4_ExpC.m4_ActionExecudeQiongju_F(m4_OriginalData, i,
#                                                                                                  m4_ActionList,
#                                                                                                  m4_ActionMatrix)
#         print('Batch:', m4_ExpC.m4_Lun, ',Stata:', m4_ExpC.m4_StateCount, ',shape:', m4_Store_np.shape)
#         m4_Store = m4_Save.m4_SaveStore_F(m4_Current, m4_Action, m4_Reward, m4_NextState, done)
#         m4_Store_np = np.vstack((m4_Store_np, m4_Store))
#
#         if m4_Store_np.shape[0] >= 2000:
#             m4_Store_np = np.delete(m4_Store_np, 0, 0)
#             m4_List.append(m4_Store_np)
#             m4_Store_np = np.zeros(413)
#
#     m4_ExpC.m4_StateCount += 1
#     if m4_ExpC.m4_StateCount >= m4_Length:
#         m4_List.append(m4_Store_np)
#         break
#
# m4_StoreAll_np = np.zeros(413)
# m4_Count = 0
# for i in m4_List:
#     m4_StoreAll_np = np.vstack((m4_StoreAll_np, i))
#     m4_Count += 1
#     print('Done!', m4_Count)
#
# # m4_Store_np = np.delete(m4_Store_np, 0, 0)
# print('Total:', m4_StoreAll_np.shape)
#
# np.savetxt(m4_ExplorePath, m4_StoreAll_np, fmt='%.6f')
# print('AllDone!')

class m4_get_experience:
    def __init__(self, cfg):
        self.DatasetDir = cfg.DatasetDir
        self.DatasetName = cfg.DatasetName
        self.SaveDir = cfg.SaveDir
        self.SaveName = cfg.SaveName
        self.num_action = cfg.num_action

    def m4_ActionExecude(self, filename, actions_set):
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

    def m4_save_experience_file(self, num_person, actions_set, DatasetDir, DatasetName, save_file_path):
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        for idx in range(1, num_person+1):
            file_name = str(idx) + '.txt'
            file_name = os.path.join(DatasetDir, DatasetName, file_name)
            experience = self.m4_ActionExecude(file_name, actions_set)
            np.savetxt(os.path.join(save_file_path, str(idx)+'_experience.txt'), experience, fmt='%.6f')
            print('Next person....')
        print('Done!')




if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--DatasetDir',default='F:/Pedestrians_Data/Non_Reward/Convection',help='near state dataset save dir')
    parse.add_argument('--DatasetName', default='Convection1/', help='near state dataset name')
    parse.add_argument('--SaveDir', default='F:/Pedestrians_Data_Processed/Non_Reward/Convection',help='experience data save dir')
    parse.add_argument('--SaveName', default='Convection1/', help='experience data set name')

    parse.add_argument('--num_action', default=180, type=int, help='number of action')
    parse.add_argument('--num_person', default=26, type=int, help='number of person')

    args = parse.parse_args()

    data = m4_get_experience(args)
    m4_ActionList = np.linspace(0, 358, args.num_action)
    data.m4_save_experience_file(args.num_person, m4_ActionList, args.DatasetDir,
                                 args.DatasetName, os.path.join(args.SaveDir, args.SaveName))
