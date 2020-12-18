# Reinforcement Learning-based Visual Navigation with Information-Theoretic Regularization
This is the implementation of our RA-L paper [arXiv](https://arxiv.org/abs/1912.04078), training and evaluation on [AI2-THOR](https://github.com/allenai/ai2thor).<br>

## Implementation
### Training
* The environment: Cuda 10.0, Python 3.6.4, PyTorch 1.0.1 
* Please install [AI2-THOR](https://github.com/allenai/ai2thor) Version 2.1.0 and modify the "controller.py" of AI2-THOR as [FILE](https://github.com/wqynew/RL-based-navigation/blob/main/change.txt).
* Our trained model can be downloaded from [HERE](https://drive.google.com/open?id=182D_0hP7orpJKyDDLlUyV4URwT3Rt0Ux). If you plan to train your own navigation model from scratch, some suggestions are provided:
    * Run the training model: python3 network.py
    * Please first run on 20 Kitchen scenes until it converges. Then increase the trining scenes to 40 (20 for kitchen, 20 for living room.) Finally, you can run on 80 training scenes.
    
### Testing
* To evaluate our model, please run "python3 eve1_checkpoint.py" 
* The files in './test/evaluation1' are for cross-scene evaluation; The files in './test/evaluation2' are for cross-target evaluation.

## Results
<div align="center">
  <table style="width:100%" border="0">
    <thead>
        <tr>
            <th>Cross-scene Generalization on AI2-THOR</th>
        </tr>
    </thead>
    <tbody>
       <tr>
         <td align="center" colspan=1><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_011_1_001110011030101_001110005720101.gif'></td>
         <td align="center" colspan=1><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310002970101_001310004330101.gif'></td>
         <td align="center" colspan=1><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_011_1_001110011030101_001110005720101.gif'></td>
         <td align="center" colspan=1><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310002970101_001310004330101.gif'></td>
       </tr>
    </tbody>
    <thead>
        <tr>
            <th>Transfer to the real world</th>
        </tr>
    </thead>
    <tbody>
       <tr>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_011_1_001110011030101_001110005720101.gif'></td>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310002970101_001310004330101.gif'></td>
       </tr>
    </tbody>
  </table>
</div>

## Contact
To ask questions or report issues please open an issue on the [issues tracker](https://github.com/wqynew/RL-based-navigation/issues).
## Citation
If you use this work in your research, please cite the paper:
```
@article{rlnav2020,
  author    = {Qiaoyun Wu and
               Kai Xu and
               Jun Wang and
               Mingliang Xu and
               Dinesh Manocha},
  title     = {Reinforcement Learning based Visual Navigation with Information-Theoretic
               Regularization},
  year      = {2019},
  url       = {http://arxiv.org/abs/1912.04078},
  archivePrefix = {arXiv},
  eprint    = {1912.04078},
}
```
