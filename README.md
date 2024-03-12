# superimposing Image Output

https://github.com/nisarg15/Superimposing_Image-AR/assets/89348092/f040d593-e08f-4594-8b23-a960db0b5b07

# AR Video


https://github.com/nisarg15/Superimposing_Image-AR/assets/89348092/0acfe8b1-25b3-4a8a-ad00-0ce72fbc1703

  
## Requirements
       ***Important****************************************************************
       *Change the path of the video in cv2.VideoCapture() function in python file*
       *Change the path of the refrence image and the testudo image in the Part1b.py, Part2a.py respectively
       *The video files are not genetrting but u can see the output in the seperate window frame by frame in a form of a video
       		
       ****************************************************************************
       
### To run this code following libraries are required
* OpenCV,  

* NumPy, 

* copy

### Installation (For ubuntu 18.04) ###
* OpenCV
	````
	sudo apt install python3-opencv
	````

* NumPy
	````
	pip install numpy
	````
	
### Running code in ubuntu
After changing the path of the video source file and installing dependencies
Make sure that current working derectory is same as the directory of program
You can change the working derectory by using **cd** command
* Run the following command which will give the result

````
python superimposing.py
````
* Run the following command which will superimpose the testudo on the AR tag.
  The code will display several images on one widow in the form of the video
````
python AR.py
````
* Run the following command which will draw the cube in 3d on the AR tag.
  The code will display several images on one widow in the form of the video
  
It is important to note that all python files are in different directory
we have to change to the correct directory again.



### Troubleshooting ###
	Most of the cases the issue will be incorrect file path.
	Double check the path by opening the properies of the video and the image of the refrence Tag and the testudo image
	and copying path directly from there.

	For issues that you may encounter create an issue on GitHub.
  
### Maintainers ###
	Nisarg Upadhyay (nisargupadhyay1@gmail.com)
