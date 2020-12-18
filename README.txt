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
