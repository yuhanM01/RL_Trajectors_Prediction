import argparse
import numpy as np
import os
import heapq

class zzt_DataProcessing():

    def __init__(self,DataPath,DataDir,AlterPath,AlterDir,SavePath,SaveDir,FrameRate,Radius,NearestNum,PeopleNum):
        self.DataPath = DataPath
        self.DataDir = DataDir
        self.AlterPath = AlterPath
        self.AlterDir = AlterDir
        self.SavePath = SavePath
        self.SaveDir = SaveDir
        self.FrameRate = FrameRate
        self.Radius = Radius
        self.NearestNum = NearestNum
        self.PeopleNum = PeopleNum

    def zzt_GetSelfState(self):

        SlefStateDir = os.path.join(self.DataPath, self.DataDir)  # 实验数据文件目录
        AlterStateDir = os.path.join(self.AlterPath, self.AlterDir)  # 初步计算后的状态文件目录
        NearStateDir = os.path.join(self.SavePath, self.SaveDir)  # 保存与周围人关系状态的文件目录

        # 如果保存目录不存在就创建保存目录
        if not os.path.exists(AlterStateDir):
            os.makedirs(AlterStateDir)

        PedList = []  # 场景中所有行人的数据
        # 读出场景中所有行人的数据
        for i in range(1, self.PeopleNum+1):
            name = SlefStateDir + str(i) + '.txt'
            file = np.loadtxt(name)
            PedList.append(file)

        # 获取所有行人的位置：Px，Py，速度：Vx，Vy，方向：Theta
        FrameStep = 8  # 隔m4_FrameStep取数据
        Count = 0
        for Agent in PedList:
            Count += 1
            AgentNum = Agent.shape[0]  # agent的数据个数
            State = []
            for i in range(AgentNum - 1):
                list_temp = []
                iStep = i + FrameStep
                if iStep <= AgentNum - 1:
                    position_hat = Agent[iStep] - Agent[i]  # / (m4_FrameRate * m4_FrameStep)
                    deta_x = Agent[i][0]  # 位置坐标x
                    deta_y = Agent[i][1]  # 位置坐标y
                    deta_px = position_hat[0] / FrameStep  # 相对位移px
                    deta_py = position_hat[1] / FrameStep  # 相对位移py
                    deta_vx = position_hat[0] / (self.FrameRate * FrameStep)
                    deta_vy = position_hat[1] / (self.FrameRate * FrameStep)

                    if deta_px == 0.0 and deta_py > 0.0:
                        theata = np.pi / 2.0

                    elif deta_px == 0.0 and deta_py < 0.0:
                        theata = np.pi / 2.0 * 3.0

                    elif deta_px > 0.0 and deta_py > 0.0:
                        theata = np.arctan(deta_py / deta_px)

                    elif deta_px < 0.0 and deta_py > 0.0:
                        theata = np.pi - np.arctan(np.abs(deta_py / deta_px))

                    elif deta_px < 0.0 and deta_py < 0.0:
                        theata = np.arctan(np.abs(deta_py / deta_px)) + np.pi

                    elif deta_px > 0.0 and deta_py < 0.0:
                        theata = np.pi * 2.0 - np.arctan(np.abs(deta_py / deta_px))

                    elif deta_px > 0.0 and deta_py == 0.0:
                        theata = 0.0

                    elif deta_px < 0.0 and deta_py == 0.0:
                        theata = np.pi
                    else:
                        theata = 0.000001
                else:
                    position_hat = Agent[AgentNum - 1] - Agent[i]  # / (m4_FrameRate * (m4_AgentNum - 1 - i))

                    deta_x = Agent[i][0]
                    deta_y = Agent[i][1]
                    deta_px = position_hat[0] / (AgentNum - 1 - i)
                    deta_py = position_hat[1] / (AgentNum - 1 - i)
                    deta_vx = position_hat[0] / (self.FrameRate * (AgentNum - 1 - i))
                    deta_vy = position_hat[1] / (self.FrameRate * (AgentNum - 1 - i))

                    if deta_px == 0.0 and deta_py > 0.0:
                        theata = np.pi / 2.0

                    elif deta_px == 0.0 and deta_py < 0.0:
                        theata = np.pi / 2.0 * 3.0

                    elif deta_px > 0.0 and deta_py > 0.0:
                        theata = np.arctan(deta_py / deta_px)

                    elif deta_px < 0.0 and deta_py > 0.0:
                        theata = np.pi - np.arctan(np.abs(deta_py / deta_px))

                    elif deta_px < 0.0 and deta_py < 0.0:
                        theata = np.arctan(np.abs(deta_py / deta_px)) + np.pi

                    elif deta_px > 0.0 and deta_py < 0.0:
                        theata = np.pi * 2.0 - np.arctan(np.abs(deta_py / deta_px))

                    elif deta_px > 0.0 and deta_py == 0.0:
                        theata = 0.0

                    elif deta_px < 0.0 and deta_py == 0.0:
                        theata = np.pi
                    else:
                        theata = 0.000001
                list_temp = [deta_x, deta_y, deta_vx, deta_vy, theata]
                State.append(list_temp)

            State_np = np.array(State, dtype=float)

            # # 隐藏这几句
            # for i in range(m4_State_np.shape[0]):
            #     m4_vDx = np.random.uniform(-np.sqrt(2), np.sqrt(2), 1)
            #     m4_vDy = np.random.uniform(-np.sqrt(2), np.sqrt(2), 1)
            #     m4_State_np[i][2] += m4_vDx
            #     m4_State_np[i][3] += m4_vDy

            name1 = AlterStateDir + str(Count) + '.txt'
            np.savetxt(name1, State_np, fmt='%.6f')
            print(str(Count)+'Done!')

    def zzt_GetNearState (self):

        SlefStateDir = os.path.join(self.DataPath, self.DataDir)  # 实验数据文件目录
        AlterStateDir = os.path.join(self.AlterPath, self.AlterDir)  # 初步计算后的状态文件目录
        NearStateDir = os.path.join(self.SavePath, self.SaveDir)  # 保存与周围人关系状态的文件目录

        PedList = []  # 场景中所有行人的数据
        PedListNew = []  # 补充成一样多的列表
        Initial = np.zeros([1, 2], dtype=float)
        MaxNumList = []

        CountUnify = 0  # 数据长度相等的文件名
        Count = 0  # 最后获取周围状态的文件名

        #self.PepoleNum = self.PepoleNum + 1
        # 读出场景中所有行人的数据
        for i in range(1, self.PeopleNum+1):
            name = AlterStateDir + str(i) + '.txt'
            file = np.loadtxt(name)
            MaxNumList.append(file.shape[0])  # 将每个行人的坐标总数添加到列表中
            PedList.append(file)#所有行人数据以矩阵的形式添加到列表中

        # 所有行人坐标的最大数目
        MaxNumList_np = np.array(MaxNumList, int)
        MaxNum = np.max(MaxNumList_np)
        #print(MaxNum)

        # 將每個人的坐標點數統一到最大那個人的坐標點數
        for AgentIndex in PedList:
            count = 1
            if AgentIndex.shape[0] < MaxNum:
                additem = AgentIndex[AgentIndex.shape[0] - 1]
                ChaE = MaxNum - AgentIndex.shape[0]
                while count <= ChaE:
                    AgentIndex = np.vstack((AgentIndex, additem))
                    count += 1
            PedListNew.append(AgentIndex)
        #print(AgentIndex.shape)

        PedListNew_np = np.array(PedListNew, float)  # 转成ndarray，所有行人状态列表

        # #将重新处理过的长度都一样的行人状态进行保存
        # for iiii in PedListNew_np:
        #     CountUnify += 1
        #     nameUnify = AlterStateDir + str(CountUnify) + '.txt'
        #     np.savetxt(nameUnify, iiii, fmt='%.6f')

        AllPerdstrainList = []  # 所有行人的距离数据
        # 计算出与附近人的距离
        for Index in range(len(PedListNew)):
            # 获取除当前行人状态其他行人状体的数据集
            PedListTemp = PedListNew.copy()
            del PedListTemp[Index]
            CurrentAgent = PedListNew[Index]
            TimeAllList = []  # 某个行人所有时间戳处，与附近行人的距离
            for TimeIndex in range(MaxNum):
                TimeList = []  # 某个行人同一个时间戳处，与附近行人的距离
                for EveryAgent in PedListTemp:
                    Distabce_hat = CurrentAgent[TimeIndex] - EveryAgent[TimeIndex]
                    Distabce_x = Distabce_hat[0]
                    Distabce_y = Distabce_hat[1]
                    Distance = np.sqrt((Distabce_x ** 2 + Distabce_y ** 2))  # 求两人之间的距离
                    TimeList.append(Distance)#某个人某时和所有人的距离
                TimeAllList.append(TimeList)#某个人所有时间和所有人的距离
            AllPerdstrainList.append(TimeAllList)#所有人所有时间的距离

        # 求与周围行人的关系
        XingrenIndex = 0
        NearestStateAll = []

        for Dim1 in AllPerdstrainList:  # 第几个行人
            NearestState = []
            TimeCount = -1
            for Dim2 in Dim1:  # 该行人某个时刻与附近所有行人的距离,Dim2中25个元素
                TimeCount += 1
                Dim2_np = np.array(Dim2, float)
                SmallestFiveAgent = heapq.nsmallest(self.NearestNum, Dim2_np)  # 获取时间戳处最近的5个人

                # 判断半径3米内是否有5个人
                LetterNum = 0
                for SmallestNumber in SmallestFiveAgent:
                    if SmallestNumber >=self.Radius:
                        LetterNum += 1

                if LetterNum == 0:  # 如果有5个人的话
                    SmallFiveIndex = heapq.nsmallest(self.NearestNum, range(len(Dim2_np)), Dim2_np.take)  # 具体是某几个行人
                    for SmallFiveIndexX in range(len(SmallFiveIndex)):  # 将下标统一到真实的数据中
                        if SmallFiveIndex[SmallFiveIndexX] >= XingrenIndex:
                            SmallFiveIndex[SmallFiveIndexX] = SmallFiveIndex[SmallFiveIndexX] + 1

                    NearFiveXingren = PedListNew_np[SmallFiveIndex]  # 最近5个行人的状态
                    CurrentHuma = PedListNew_np[XingrenIndex]  # 当前的那个行人状态
                    NearestStateTime = NearFiveXingren - CurrentHuma

                    Px = CurrentHuma[TimeCount][0]
                    Py = CurrentHuma[TimeCount][1]
                    Vx = CurrentHuma[TimeCount][2]
                    Vy = CurrentHuma[TimeCount][3]
                    Theta = CurrentHuma[TimeCount][4]

                    P0x = NearestStateTime[0][TimeCount][0]
                    P1x = NearestStateTime[1][TimeCount][0]
                    P2x = NearestStateTime[2][TimeCount][0]
                    P3x = NearestStateTime[3][TimeCount][0]
                    P4x = NearestStateTime[4][TimeCount][0]

                    P0y = NearestStateTime[0][TimeCount][1]
                    P1y = NearestStateTime[1][TimeCount][1]
                    P2y = NearestStateTime[2][TimeCount][1]
                    P3y = NearestStateTime[3][TimeCount][1]
                    P4y = NearestStateTime[4][TimeCount][1]

                    V0x = NearestStateTime[0][TimeCount][2]
                    V1x = NearestStateTime[1][TimeCount][2]
                    V2x = NearestStateTime[2][TimeCount][2]
                    V3x = NearestStateTime[3][TimeCount][2]
                    V4x = NearestStateTime[4][TimeCount][2]

                    V0y = NearestStateTime[0][TimeCount][3]
                    V1y = NearestStateTime[1][TimeCount][3]
                    V2y = NearestStateTime[2][TimeCount][3]
                    V3y = NearestStateTime[3][TimeCount][3]
                    V4y = NearestStateTime[4][TimeCount][3]

                    NearestStateTimeList = [Px, Py, Vx, Vy, Theta,
                                               P0x, P1x, P2x, P3x, P4x,
                                               P0y, P1y, P2y, P3y, P4y,
                                               V0x, V1x, V2x, V3x, V4x,
                                               V0y, V1y, V2y, V3y, V4y]

                    # m4_NearestState.append(m4_NearestStateTimeList)


                else:  # 不足5个人，补0
                    SmallFiveIndex = heapq.nsmallest((self.NearestNum - LetterNum), range(len(Dim2_np)),
                                                        Dim2_np.take)  # 具体是某几个行人

                    for SmallFiveIndexX in range(len(SmallFiveIndex)):  # 将下标统一到真实的数据中
                        if SmallFiveIndex[SmallFiveIndexX] >= XingrenIndex:
                            SmallFiveIndex[SmallFiveIndexX] = SmallFiveIndex[SmallFiveIndexX] + 1

                    if len(SmallFiveIndex) > 0:
                        NearFiveXingren = PedListNew_np[SmallFiveIndex]  # 最近5个行人的状态
                        CurrentHuma = PedListNew_np[XingrenIndex]  # 当前的那个行人状态
                        NearestStateTime = NearFiveXingren - CurrentHuma

                        Px = CurrentHuma[TimeCount][0]
                        Py = CurrentHuma[TimeCount][1]
                        Vx = CurrentHuma[TimeCount][2]
                        Vy = CurrentHuma[TimeCount][3]
                        Theta = CurrentHuma[TimeCount][4]

                        YouPxList = []
                        YouPyList = []
                        YouVxList = []
                        YouVyList = []
                        for YouIndex in range(len(SmallFiveIndex)):
                            YouPx = NearestStateTime[YouIndex][TimeCount][0]
                            YouPy = NearestStateTime[YouIndex][TimeCount][1]
                            YouVx = NearestStateTime[YouIndex][TimeCount][2]
                            YouVy = NearestStateTime[YouIndex][TimeCount][3]

                            YouPxList.append(YouPx)
                            YouPyList.append(YouPy)
                            YouVxList.append(YouVx)
                            YouVyList.append(YouVy)

                        for JiaIndex in range(LetterNum):
                            YouPxList.append(0.0)
                            YouPyList.append(0.0)
                            YouVxList.append(0.0)
                            YouVyList.append(0.0)

                        NearestStateTimeList = [Px, Py, Vx, Vy, Theta]

                        for zhuanIndex in range(self.NearestNum):
                            NearestStateTimeList.append(YouPxList[zhuanIndex])

                        for zhuanIndex in range(self.NearestNum):
                            NearestStateTimeList.append(YouPyList[zhuanIndex])

                        for zhuanIndex in range(self.NearestNum):
                            NearestStateTimeList.append(YouVxList[zhuanIndex])

                        for zhuanIndex in range(self.NearestNum):
                            NearestStateTimeList.append(YouVyList[zhuanIndex])

                        # m4_NearestState.append(m4_NearestStateTimeListZ)



                    else:
                        CurrentHuma = PedListNew_np[XingrenIndex]  # 当前的那个行人状态
                        Px = CurrentHuma[TimeCount][0]
                        Py = CurrentHuma[TimeCount][1]
                        Vx = CurrentHuma[TimeCount][2]
                        Vy = CurrentHuma[TimeCount][3]
                        Theta = CurrentHuma[TimeCount][4]
                        NearestStateTimeList = [Px, Py, Vx, Vy, Theta,
                                                   0.0, 0.0, 0.0, 0.0, 0.0,
                                                   0.0, 0.0, 0.0, 0.0, 0.0,
                                                   0.0, 0.0, 0.0, 0.0, 0.0,
                                                   0.0, 0.0, 0.0, 0.0, 0.0]
                # print(len(m4_NearestStateTimeList))

                NearestState.append(NearestStateTimeList)

            NearestStateAll.append(NearestState)
            XingrenIndex += 1
        NearestStateAll_np = np.array(NearestStateAll, float)

        # m4_AStarList = []
        # for AStar in range(1,m4_PepoleNum):
        #     AstarPath = m4_TongyongPath + '/AStar/行人结果' +str(AStar) +'.txt'
        #     AStarFile = np.loadtxt(AstarPath)
        #     m4_AStarList.append(AStarFile)
        #     print(AStarFile.shape)

        #如果保存目录不存在就创建保存目录
        if not os.path.exists(NearStateDir ):
            os.makedirs(NearStateDir)

        iiiiindex = 0
        # 保存与附近人的状态文件
        for iii in NearestStateAll_np:
            length = MaxNumList[iiiiindex]
            iii = iii[0:length - 1]
            # np.insert(iii,25,values=m4_AStarList[iiiiindex],axis=1)
            iiiiindex += 1
            Count += 1
            name1 = NearStateDir + str(Count) + '.txt'
            np.savetxt(name1, iii, fmt='%.6f')
            print('Done:', Count)
            print('Ped', iii.shape)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--DataPath',default='F:/Pedestrians_Data/Non_Reward/Convection',help='The path of the experimental data')
    parse.add_argument('--DataDir', default='Convection1/', help='The directory of the experimental data')
    parse.add_argument('--AlterPath', default='F:/Pedestrians_Data_Alter/Non_Reward/Convection', help='The path of the alter data')
    parse.add_argument('--AlterDir', default='Convection1/', help='The directory of the alter data')
    parse.add_argument('--SavePath', default='F:/Pedestrians_Data_Processed/Non_Reward/Convection',help='The path to save the processed data')
    parse.add_argument('--SaveDir', default='Convection1/', help='The directory to save the processed data')
    parse.add_argument('--FrameRate', default=1.0/30.0,type=float, help='The time interval between each frame')
    parse.add_argument('--Radius', default=3.0, type=float, help='The search radius')
    parse.add_argument('--NearestNum', default=5, type=int, help='The number of neighbors to consider')
    parse.add_argument('--PeopleNum', default=26, type=int, help='The number of pedestrians')
    args = parse.parse_args()

    data = zzt_DataProcessing(args.DataPath,args.DataDir,args.AlterPath,args.AlterDir,args.SavePath,args.SaveDir,
                       args.FrameRate,args.Radius,args.NearestNum,args.PeopleNum)
    data.zzt_GetSelfState()
    data.zzt_GetNearState()
    print('hahahah')