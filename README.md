# __行人轨迹仿真__
1. ***数据集介绍与转化成`.tfrecoders`文件***
    1. ***数据集介绍***  
    主要是疏散和对流两个场景，分为有竞争，无竞争，数据集[下载地址](https://pan.baidu.com/s/1CeBN5ZtGVWRH8BQeCqkvfg), 提取码：`i5ic`  
    `.txt`文件存储的数据格式为：每行存储````x, y ````这样的位置坐标  
    数据集的坐标系如下图所示：  
    ![image](https://github.com/yuhanM01/RL_Trajectors_Prediction/blob/master/image/%E5%9C%BA%E6%99%AF%E5%9B%BE.png)
    2. ***转化成`.tfrecoders`***  
        1. 生成行人与周围人的状态：  
            这一操作会生成两个文件夹：  
            1）存储行人自身状态的文件夹：`.txt`文件中存储的为行人自身的`位置`，`速度`和`方向(单位：弧度)`。[x, y, vx, vy, Direction]  
            2）存储行人与周围人状态的文件夹：`.txt`文件中存储的为行人与周为n个人的`自身的状态`，`相对位置`，`相对速度` .[Px, Py, Vx, Vy, Theta, P0x, P1x, P2x, P3x, P4x, P0y, P1y, P2y, P3y, P4y, V0x, V1x, V2x, V3x, V4x, V0y, V1y, V2y, V3y, V4y]
            生成方法如下：  
            1. 找到[m4_GetState.sh](https://github.com/yuhanM01/RL_Trajectors_Prediction/blob/master/m4_GetState.sh)，修改对应参数即可  
            举个例子： 
                ````
                python zzt_GetNearState_File.py \
                --DataPath='/home/yang/study/datasetandparam/Predestrain_dataset/non-comp/V-Z' \ 数据集的目录
                --DataDir='1/' \数据集的名称
                --AlterPath='/home/yang/study/datasetandparam/Predestrain_dataset/non-comp/V-Z' \ 自身状态存储的目录
                --AlterDir='1_selfstate/' \ 自身状态存储的文件夹
                --SavePath='/home/yang/study/datasetandparam/Predestrain_dataset/non-comp/V-Z' \ 与周围状态存储的目录
                --SaveDir='1_nearstate/' \ 与周围状态存储的文件夹
                --FrameRate=0.033333333 \ 帧率
                --Radius=3.0 \ 半径3米的人
                --NearestNum=5 \ 这个参数只能是5
                --PeopleNum=26 数据集中有多少个行人
                ````  
        2. 生成经验文件：  
            这一操作会生成两个文件夹：  
            1）存储行人`方向经验`的文件夹：`.txt`文件中存储的为[CurrentState, action, m4_Reward, m4_NextState, is_done], 每行[25+1+1+25+1]个数  
            2）存储行人`速度大小经验`的的文件夹：`.txt`文件中存储的为[CurrentState, action, m4_Reward, m4_NextState, is_done], 每行[25+1+1+25+1]个数  
            生成方法如下：  
            找到[m4_get_experience.sh](https://github.com/yuhanM01/RL_Trajectors_Prediction/blob/master/m4_get_experoence.sh)文件  
            举个例子：  
            
                python m4_get_experience.py \
                --DatasetDir='/home/yang/study/datasetandparam/Predestrain_dataset/comp/counterflow' \ 周围状态数据集所在的目录
                --DatasetName='1_1_nearstate' \ 周围状态数据集的名称
                --SaveDirecitonDir='/home/yang/study/datasetandparam/Predestrain_dataset/comp/counterflow' \ 保存方向经验数据集所在的目录
                --SaveDirecitonName='1_1_DirectionExperience' \ 保存方向经验数据集所在的名称
                --SaveVelocityDir='/home/yang/study/datasetandparam/Predestrain_dataset/comp/counterflow' \ 保存方速度大小验数据集所在的目录
                --SaveVelocityName='1_1_VelocityExperience' \ 保存方速度大小验数据集所在的名称
                --num_DirectionAction=180 \
                --num_VelocityAction=201 \
                --num_person=26
        3. 转化成`.tfrecoders`  
        生成方法如下：
        找到[m4_make_tfrecord.sh](https://github.com/yuhanM01/RL_Trajectors_Prediction/blob/master/m4_make_tfrecord.sh)文件    
        举个例子：
            ````
            python m4_make_tfrecord.py \
            --SenceName='Counterflow_Comp' \ 场景的名称，因为要将所有场景生成的tfrecord文件拷到同一个文件夹，所以名称要不一样，加个场景名称区分
            --DirectionDatasetDir='/home/yang/study/datasetandparam/Predestrain_dataset/comp/counterflow' \ 生成的经验数据集.txt的目录
            --DirectionDatasetName='1_1_DirectionExperience' \ 生成的经验数据集.txt的名称
            --SaveDirecitonDir='/home/yang/study/datasetandparam/Predestrain_dataset/comp/counterflow' \ 保存生成的tfrecord文件目录
            --SaveDirecitonName='1_1_DirectionTfrecord' \ 保存生成的tfrecord文件名称
            --VelocityDatasetDir='/home/yang/study/datasetandparam/Predestrain_dataset/comp/counterflow' \ 生成的经验数据集.txt的目录
            --VelocityDatasetName='1_1_VelocityExperience' \ 生成的经验数据集.txt的名称
            --SaveVelocityDir='/home/yang/study/datasetandparam/Predestrain_dataset/comp/counterflow' \ 保存生成的tfrecord文件目录
            --SaveVelocityName='1_1_VelocityTfrecord' \ 保存生成的tfrecord文件名称
            --num_person=26
            ````
