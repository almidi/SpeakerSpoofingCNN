from lib.model_io import *
from src.train import train
from src.predict import predict



# Caution ! Train one at a time !!
########################################################################################################################
# Train Model
dict = train(epochs = 64,model='baseline',early_stop=6,learning_rate=0.001,batch_norm=True,batch_size=256)
# Predict runs on last model
dict['accuracy'] = predict(epochs = 64,model='baseline',early_stop=6,learning_rate=0.001,batch_norm=True,batch_size=256)
# Save data
save_model_stats(dict.get('model_id'),dict)

# ########################################################################################################################
# # Train Model
# dict = train(epochs = 64,model='vd10fd',early_stop=12,learning_rate=0.001,batch_norm=False,batch_size=256)
# # Predict runs on last model
# dict['accuracy'] = predict(epochs = 4,model='vd10fd',early_stop=12,learning_rate=0.001,batch_norm=True,batch_size=256)
# # Save data
# save_model_stats(dict.get('model_id'),dict)

# ########################################################################################################################
# # Train Model
# dict = train(epochs = 4,model='vd10fd',early_stop=12,learning_rate=0.001,batch_norm=True,batch_size=256)
# # Predict runs on last model
# dict['accuracy'] = predict(epochs = 4,model='vd10fd',early_stop=12,learning_rate=0.001,batch_norm=True,batch_size=256)
# # Save data
# save_model_stats(dict.get('model_id'),dict)

# ########################################################################################################################
# # Train Model
# dict = train(epochs = 4,model='vd10fd',early_stop=12,learning_rate=0.001,batch_norm=False,batch_size=256)
# # Predict runs on last model
# dict['accuracy'] = predict(epochs = 4,model='vd10fd',early_stop=12,learning_rate=0.001,batch_norm=True,batch_size=256)
# # Save data
# save_model_stats(dict.get('model_id'),dict)
