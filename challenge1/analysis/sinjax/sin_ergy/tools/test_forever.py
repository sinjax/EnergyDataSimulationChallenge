from sin_ergy.data.utils import load_training, load_test
from sin_ergy.models import DeepConvEnergy
from sin_ergy.tools.evaluate_all_models import NO_TIME
import numpy as np
cols = NO_TIME
print("Starting to train deepconvenergy on real data...")
seen = []
while True:
    xtrain, ytrain= load_training(cols,sequence=True,window_size=12)
    xtest, ytest= load_test(cols,sequence=True,window_size=12)

    model = DeepConvEnergy(12,len(cols),nfilters=30,filterwidth=3)
    model.compile(optimizer="adam", loss="mape")
    model.fit(xtrain,ytrain[:,-1,:],nb_epoch=15,verbose=0)
    eval_mape = model.evaluate(xtest, ytest[:, -1, :],verbose=0)
    seen += [eval_mape]
    print("Mean: %2.2f, Std: %2.2f"%(np.mean(seen),np.std(seen)))