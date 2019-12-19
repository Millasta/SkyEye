from program.Deep.unet.preprocessing import *
from program.Deep.unet.model import *

#maskFusion()

data_gen_args = dict(rotation_range=360,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='reflect')

myGenerator = trainGenerator(10,'data/','img','classes',data_gen_args,save_to_dir = "data/aug")

#num_batch = 1
#for i,batch in enumerate(myGenerator):
#    print(str(i*10) + "%")
#    if(i >= num_batch):
#        break
#print("Aug done !")

model = unet()
model_checkpoint = ModelCheckpoint('unet_lidar.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGenerator,steps_per_epoch=1,epochs=5,callbacks=[model_checkpoint])

