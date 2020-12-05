from src import *
from src import datasets1, datasets2

from src import models_N1,networks_N1
#from src import models_N2,networks_N2
#from src import models_N3,networks_N3
#from src import models_P4,networks_P4
#from src import models_P5,networks_P5
#from src import models_P6,networks_P6

#from src import models1_N1,networks1_N1
#from src import models1_N2,networks1_N2
#from src import models1_P4,networks1_P4
#from src import models1_P5,networks1_P5
#from src import models1_P6,networks1_P6
#from src import models1_P8,networks1_P8

#from src import models2_N1,networks2_N1
#from src import models2_N2,networks2_N2
#from src import models2_N3,networks2_N3
#from src import models2_P4,networks2_P4
#from src import models2_P5,networks2_P5
#from src import models2_P6,networks2_P6

if __name__ == '__main__':
    args = ModelOptions().parse()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
 
    pw_network_N1 = networks_N1.PatchWiseNetwork(args.channels)
    iw_network_N1 = networks_N1.ImageWiseNetwork(args.channels)
    
    #pw_network_N2 = networks_N2.PatchWiseNetwork(args.channels)
    #iw_network_N2 = networks_N2.ImageWiseNetwork(args.channels)
    
    #pw_network_N3 = networks_N3.PatchWiseNetwork(args.channels)
    #iw_network_N3 = networks_N3.ImageWiseNetwork(args.channels)
    
    #pw_network_P4 = networks_P4.PatchWiseNetwork(args.channels)
    #iw_network_P4 = networks_P4.ImageWiseNetwork(args.channels)

    #pw_network_P5 = networks_P5.PatchWiseNetwork(args.channels)
    #iw_network_P5 = networks_P5.ImageWiseNetwork(args.channels)
    
    #pw_network_P6 = networks_P6.PatchWiseNetwork(args.channels)
    #iw_network_P6 = networks_P6.ImageWiseNetwork(args.channels)
     
    #--------------------------------------------------------------
    
    #pw_network1_N1 = networks1_N1.PatchWiseNetwork1(args.channels)
    #iw_network1_N1 = networks1_N1.ImageWiseNetwork1(args.channels)
    
    #pw_network1_N2 = networks1_N2.PatchWiseNetwork1(args.channels)
    #iw_network1_N2 = networks1_N2.ImageWiseNetwork1(args.channels)
    
    #pw_network1_P4 = networks1_P4.PatchWiseNetwork1(args.channels)
    #iw_network1_P4 = networks1_P4.ImageWiseNetwork1(args.channels)
    
    #pw_network1_P5 = networks1_P5.PatchWiseNetwork1(args.channels)
    #iw_network1_P5 = networks1_P5.ImageWiseNetwork1(args.channels)
    
    #pw_network1_P6 = networks1_P6.PatchWiseNetwork1(args.channels)
    #iw_network1_P6 = networks1_P6.ImageWiseNetwork1(args.channels)

    #pw_network1_P8 = networks1_P8.PatchWiseNetwork1(args.channels)
    #iw_network1_P8 = networks1_P8.ImageWiseNetwork1(args.channels)
    
    #--------------------------------------------------------------
    
    #pw_network2_N1 = networks2_N1.PatchWiseNetwork2(args.channels)
    #iw_network2_N1 = networks2_N1.ImageWiseNetwork2(args.channels)
    
    #pw_network2_N2 = networks2_N2.PatchWiseNetwork2(args.channels)
    #iw_network2_N2 = networks2_N2.ImageWiseNetwork2(args.channels)
    
    #pw_network2_N3 = networks2_N3.PatchWiseNetwork2(args.channels)
    #iw_network2_N3 = networks2_N3.ImageWiseNetwork2(args.channels)
    
    #pw_network2_P4 = networks2_P4.PatchWiseNetwork2(args.channels)
    #iw_network2_P4 = networks2_P4.ImageWiseNetwork2(args.channels)
    
    #pw_network2_P5 = networks2_P5.PatchWiseNetwork2(args.channels)
    #iw_network2_P5 = networks2_P5.ImageWiseNetwork2(args.channels)
    
    #pw_network2_P6 = networks2_P6.PatchWiseNetwork2(args.channels)
    #iw_network2_P6 = networks2_P6.ImageWiseNetwork2(args.channels)
    
    
    if args.network == '0' or args.network == '1':
        pw_model = models_N1.PatchWiseModel(args, pw_network_N1)
        pw_model.train()
        
        pw_model1 = models1_N1.PatchWiseModel1(args, pw_network1_N1)
        pw_model1.train()
        
        #pw_model2 = models2_N1.PatchWiseModel2(args, pw_network2_N1)
        #pw_model2.train()
        
   
    
    if args.network == '0' or args.network == '2':
              
        iw_model_N1 = models_N1.ImageWiseModel(args, iw_network_N1, pw_network_N1)
        iw_model_N1.train()
        
        #iw_model_N2 = models_N2.ImageWiseModel(args, iw_network_N2, pw_network_N2)
        #iw_model_N2.train()
        
        #iw_model_N3 = models_N3.ImageWiseModel(args, iw_network_N3, pw_network_N3)
        #iw_model_N3.train()
        
        #iw_model_P4 = models_P4.ImageWiseModel(args, iw_network_P4, pw_network_P4)
        #iw_model_P4.train()
        
        #iw_model_P5 = models_P5.ImageWiseModel(args, iw_network_P5, pw_network_P5)
        #iw_model_P5.train()
        
        #iw_model_P6 = models_P6.ImageWiseModel(args, iw_network_P6, pw_network_P6)
        #iw_model_P6.train()
        #---------------------------------------------------------------------------#
        
        #iw_model1_N1 = models1_N1.ImageWiseModel1(args, iw_network1_N1, pw_network1_N1)
        #iw_model1_N1.train()
        
        #iw_model1_N2 = models1_N2.ImageWiseModel1(args, iw_network1_N2, pw_network1_N2)
        #iw_model1_N2.train()
        
        #iw_model1_P4 = models1_P4.ImageWiseModel1(args, iw_network1_P4, pw_network1_P4)
        #iw_model1_P4.train()
        
        #iw_model1_P5 = models1_P5.ImageWiseModel1(args, iw_network1_P5, pw_network1_P5)
        #iw_model1_P5.train()
        
        #iw_model1_P6 = models1_P6.ImageWiseModel1(args, iw_network1_P6, pw_network1_P6)
        #iw_model1_P6.train()
        
        #iw_model1_P8 = models1_P8.ImageWiseModel1(args, iw_network1_P8, pw_network1_P8)
        #iw_model1_P8.train()
        
        #---------------------------------------------------------------------------#
        
        #iw_model2_N1 = models2_N1.ImageWiseModel2(args, iw_network2_N1, pw_network2_N1)
        #iw_model2_N1.train()
        
        #iw_model2_N2 = models2_N2.ImageWiseModel2(args, iw_network2_N2, pw_network2_N2)
        #iw_model2_N2.train()
        
        #iw_model2_N3 = models2_N3.ImageWiseModel2(args, iw_network2_N3, pw_network2_N3)
        #iw_model2_N3.train()
        
        #iw_model2_P4 = models2_P4.ImageWiseModel2(args, iw_network2_P4, pw_network2_P4)
        #iw_model2_P4.train()        
    
        #iw_model2_P5 = models2_P5.ImageWiseModel2(args, iw_network2_P5, pw_network2_P5)
        #iw_model2_P5.train()
        
        #iw_model2_P6 = models2_P6.ImageWiseModel2(args, iw_network2_P6, pw_network2_P6)
        #iw_model2_P6.train()
        
