import pandas as pd
from sklearn.model_selection import KFold

class Validation:
    def __init__(self,data):
        self.data = data
        
    def cross_validate(self,session,split_size=5):
        results = []
        kf = KFold(n_splits=split_size)
        for train_idx, val_idx in kf.split(train_x_all, train_y_all):
            train_x = train_x_all[train_idx]
            train_y = train_y_all[train_idx]
            val_x = train_x_all[val_idx]
            val_y = train_y_all[val_idx]
            run_train(session,train_x,train_y)
            results.append(session.run(accuracy, feed_dict={x: val_x, y: val_y}))
        return results
