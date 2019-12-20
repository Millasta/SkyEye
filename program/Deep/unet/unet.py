from program.Deep.unet.preprocessing import *
from program.Deep.unet.model_multi import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

#maskFusion()

def GenerateData(batch_size, save_to=None):
    data_gen_args = dict(rotation_range=360,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect')

    # num_batch = 1
    # for i,batch in enumerate(myGenerator):
    #    print(str(i*10) + "%")
    #    if(i >= num_batch):
    #        break
    # print("Aug done !")

    return trainGenerator(batch_size,'data/','img','classes',data_gen_args,save_to_dir = save_to)

def Train(generator, steps_nb, epoch_nb):
    model = unet()
    model_checkpoint = ModelCheckpoint('unet_lidar.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(generator,steps_per_epoch=steps_nb,epochs=epoch_nb,callbacks=[model_checkpoint])

def Predict():
    testGene = testGenerator("data/test")
    model = unet()
    model.load_weights("unet_lidar.hdf5")
    results = model.predict_generator(testGene,2,verbose=1)
    saveResult("data/test",results)


#maskFusion()

#generator = GenerateData(10)
#Train(generator, 1, 5)
Predict()
