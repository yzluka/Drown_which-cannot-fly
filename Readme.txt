Description: This Repo aimed in providing toolchain for simulating output result provided by neural network.

Default setting: the default setting of this programe will generate 2000*2000 pixle image with different features on it.

Please read the comment in the files to see which parameters are avilable for changes.

File explanation (By execution sequence): 

    1. Generator.py: Generating the positions, size and rotation angle of all the feature on the simulated map and then draw them out.
    
        Output: GT-testing1.png (the ground truth image)
        
                testing1.png (the high resolution image with artifically added inaccuracy)
                
    2. blurry.py: Taking the two inputs from Generator.py and add salt and pepper noise w/ CONTROLABLE intensity and proportion to testing1.png. 
    
        Output: GT-testing1_bw.png (binary map with tells if each pixel belongs to a real taget or not)
    
                testing1_blurred.png (Added salt and pepper noise to testing1.png) 
    
    3. Interprester.py: Calculating the information within each region(smaller square) and outputing the unnormalized information map. Also adding the gaussian noise to it.
    
        Output: InfoMap.npy (The information map, an 2D array which contains the information of each region WITHOUT the gaussiance noise)
                
                InfoMap_blurred.npy (The information map, an 2D array which contains the information of each region WITH the gaussiance noise)
                
                Note: Both of the .npy files take the salt and pepper noise into account and both gives UNNORMALIZED information. Normalization can be made directly on it. 
     
     4. Reconstruct.py: Visualization of the information map. Can be used to compare with the result given by GT-testing_bw.png. We can visually find that we have some ROI being
        hidden and some False positive (FP) start to appear on the product map. This fits exactly with the nature of a typical output from a Neural Network.
     
            
