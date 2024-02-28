# LLM4Rec_User_Summary
Use LLM to build Conv RecSys. 


## important notes

When testing the model to see if it predicts all items correctly remember the model is only trained on ratings that are larger than four. Therefore the naive training recall is much worse. Need to compare to those that are not naive i.e larger than four


## Ideas for 
Should consider where to go next. This is a baseline interpretation but how can you make this more autonomous a better agent for discory. Probably going to have to take a more conversational approach like the one olivier was suggesting


## Experiments 

To run aany experiment simply run 

```
pythom -m  experiment.FILE
```