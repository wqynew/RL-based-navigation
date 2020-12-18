# Reinforcement Learning-based Visual Navigation with Information-Theoretic Regularization
This is the implementation of our RA-L paper, training and evaluation on AI2-THOR.<br>
## Navigation Model
![](https://github.com/wqynew/NeoNav/raw/master/image/overview.png)
## Implementation
### Training
* The environment: Cuda 10.0, Python 3.6.4, PyTorch 1.0.1 
* Please download "depth_imgs.npy" file from the [AVD_Minimal](https://drive.google.com/file/d/1SmA-3cGwV12XKdGYdsBEJwxf1MYdE6-y/view?usp=sharing) and put the file in the train folder. 
* Please download our training data [HERE](https://drive.google.com/open?id=1Avl5CNn-V4Fpfhn0nE9siJMkYZRczKmN).
* Our trained model can be downloaded from [HERE](https://drive.google.com/open?id=182D_0hP7orpJKyDDLlUyV4URwT3Rt0Ux). If you plan to train your own navigation model from scratch, some suggestions are provided:
    * Pre-train the model by using "python3 ttrain.py" and terminate the training when the action prediction accuracy approaches 70%.
    * Use "python3 train.py" to train the NeoNav model.
    
### Testing
* To evaluate our model, please run "python3 ./test/evaluate.py" or "python3 ./test/evaluete_with_stop.py"

## Results
<div align="center">
  <table style="width:100%" border="0">
    <thead>
        <tr>
            <th>Start</th>
            <th>End</th>
            <th>Start</th>
            <th>End</th>
        </tr>
    </thead>
    <tbody>
       <tr>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s1.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t1.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s3.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t3.png'></td>
       </tr>
       <tr>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_011_1_001110011030101_001110005720101.gif'></td>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310002970101_001310004330101.gif'></td>
       </tr>
       <tr>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s2.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t2.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s4.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t4.png'></td>
       </tr>
       <tr>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310007440101_001310000150101.gif'></td>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_016_1_001610000060101_001610004220101.gif'></td>
       </tr>
    </tbody>
  </table>
</div>

## Contact
To ask questions or report issues please open an issue on the [issues tracker](https://github.com/wqynew/NeoNav/issues).
## Citation
If you use NeoNav in your research, please cite the paper:
```
@article{wuneonav,
  title={NeoNav: Improving the Generalization of Visual Navigation via Generating Next Expected Observations},
  author={Wu, Qiaoyun and Manocha, Dinesh and Wang, Jun and Xu, Kai}
}
```



(1) This version is for AI2-THOR  Version 2.1.0  (https://github.com/allenai/ai2thor)
(2) Our environment: cuda 10.0, Python 3.6.4 
(3) First, install ai2thor. 
Second, modify the "controller.py" of AI2-THOR as follows:
(/scratch1/anaconda3/lib/python3.6/site-packages/ai2thor/controller.py)
For class BFSController:
a) add function:
    def getgoal(self,idx):
        self.goal={}
        if idx in range(1,31):
            self.gkeys=["Toaster","Microwave","Fridge","CoffeeMaker","GarbageCan","Box","Bowl",'Apple', 'Chair', 'DiningTable', 'Plate', 'Sink', 'SinkBasin']
            #self.gkeys=["StoveBurner", "Cabinet","HousePlant"]
        elif idx in range(201,231):
            self.gkeys=["Pillow","Laptop","Television", "GarbageCan", "Box", "Bowl",'Book', 'FloorLamp', 'Painting', 'Sofa']
            #self.gkeys=["Statue","HousePlant","TableTop"]
        elif idx in range(301,331):
            self.gkeys=["Lamp", "Book", "AlarmClock",'Bed', 'Mirror', 'Pillow', 'GarbageCan', 'TissureBox']
            #self.gkeys=["Cabinet","Statue","Dresser","LightSwitch"]
        elif idx in range(401,431):
            self.gkeys=["Sink", "ToiletPaper", "SoapBottle", "LightSwitch",'Candle', 'GarbageCan', 'SinkBasin', 'ScrubBrush']
            #self.gkeys=["Cabinet","Towel","TowelHolder"]
        for ele in self.gkeys:
            self.goal[ele]=[]
b) modify function:
    def search_all_closed(self, scene_name):
        self.scene_name=scene_name
        self.count=0
        widx=[int(s) for s in scene_name.split('n') if s.isdigit()]
        self.getgoal(widx[0]) 
        self.allow_enqueue = True
        self.queue = deque()
        self.seen_points = []
        self.visited_seen_points = []
        self.grid_points = []
        event = self.reset(scene_name)
        event = self.step(dict(action='Initialize', renderDepthImage=True, gridSize=self.grid_size))
        self.enqueue_points(event.metadata['agent']['position'])
        while self.queue:
            self.queue_step()
        self.prune_points()

    def queue_step(self):
        search_point = self.queue.popleft()
        event = self.step(dict(
            action='Teleport',
            x=search_point.start_position['x'],
            y=search_point.start_position['y'],
            z=search_point.start_position['z']))
        #print(event.metadata['agent']['position'])
        if not event.metadata['lastActionSuccess']:
            return
        assert event.metadata['lastActionSuccess']
        move_vec = search_point.move_vector
        move_vec['moveMagnitude'] = self.grid_size
        event = self.step(dict(action='Move', renderDepthImage=True, **move_vec))

        if event.metadata['lastActionSuccess']:
            if event.metadata['agent']['position']['y'] > 1.3:
                #pprint(search_point.start_position)
                #pprint(search_point.move_vector)
                #pprint(event.metadata['agent']['position'])
                raise Exception("**** got big point ")

            self.enqueue_points(event.metadata['agent']['position'])

            if not any(map(lambda p: distance(p, event.metadata['agent']['position']) < self.distance_threshold, self.grid_points)):
                self.grid_points.append(event.metadata['agent']['position'])
                for i in range(4):
                    event=self.step(dict(action='RotateRight'))
                    for obj in event.metadata['objects']:
                        if obj['objectType'] in self.gkeys:
                            if obj['visible']: 
                                rot=event.metadata['agent']['rotation']
                                pos=event.metadata['agent']['position']
                                self.goal[obj['objectType']].append({'position':pos,'rotation':rot,'distance':obj['distance'],'frame':event.frame,'depth_frame':event.depth_frame})   
        return event
c) Run the training model: python3 network.py
Please first run on $20$ Kitchen scenes until it converges. 
Then increase the trining scenes to $40$ (20 for kitchen, 20 for living room.)
Finally, you can run on $80$ training scenes.

(4) python3 eve1_checkpoint.py for evaluation

(5) The files in './test/evaluation1' are for cross-scene evaluation; The files in './test/evaluation2' are for cross-target evaluation.

(6) We will provide our trained model parameter later (since it is too big). 
