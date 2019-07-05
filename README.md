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
                --DataPath='/home/yang/study/datasetandparam/Predestrain_dataset/non-comp/V-Z' \
                --DataDir='1/' \
                --AlterPath='/home/yang/study/datasetandparam/Predestrain_dataset/non-comp/V-Z' \
                --AlterDir='1_selfstate/' \
                --SavePath='/home/yang/study/datasetandparam/Predestrain_dataset/non-comp/V-Z' \
                --SaveDir='1_nearstate/' \
                --FrameRate=0.033333333 \
                --Radius=3.0 \ 
                --NearestNum=5 \ 这个参数只能是5
                --PeopleNum=26```` 
