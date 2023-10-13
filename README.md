
# Source code for "Improving Contrastive Learning in Emotion Recognition in Conversation via Novel Data Augmentation and Decoupling Neutral Emotion"

---
- command:
  
  ```
    python train.py --dataset [MELD/EMORY/iemocap/dailydialog] --batch 8 --loss [ce/supcon/neu] --augment centroid --prob 0.5
  ```

  ## Key argument
  
  - ```dataset```: MELD / EMORY / iemocap / dailydialog
  - ```loss```: ce(cross-entropy) / supcon(supervised contrastive learning) / neu(Our Neu-CL)
  - ```augment```: centroid (Our EmoMix) / del / swap / insert / replace / dropout
  - ```prob```: the parameter to adjust the force for neutral 
  - ```layer_set``` : set the last layer (= 23) by default 
