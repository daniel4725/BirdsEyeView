Introducing here a small taste of the project. For more detailed information see the report.pdf file

# Birds Eye View

The problem of obtaining an overhead view, also known as bird’s eye view
(BEV), is a popular problem in computer vision due to its usefulness. The
idea of transforming an image to an overhead view of a scene is utilized in applications that require a complete understanding of the surroundings. One such
application is in the field of Advanced Driving Assistance Systems (ADAS)
where the private driver uses the overhead view in tasks such as parking, navigating, and more.
Another application is remote-operated machines in worksites, where the operator operates a machine from a distance using a camera’s feed thus removing
the risks involved in operating such a machine on site. Cameras positioned
around the site are used in the composition of a bird’s eye view of the site.
This global view will serve as a virtual map used to help the operator adequately specify the commands to be sent to the vehicle under his control [2].
Additionally, an overhead view provides a better understanding of the scene
geometry, and measurements can be taken directly from the image. Hence, it
can be used as a pre-processing step for many other computer vision tasks like
object detection and tracking.

# Method
## Step A: Detecting vanishing point

![image](https://github.com/daniel4725/BirdsEyeView/assets/95569050/befc9a32-e857-4dc7-a9c1-2806f525f3d7)

## Step B: Calculating and applying the homography

![image](https://github.com/daniel4725/BirdsEyeView/assets/95569050/8ae18468-7a19-4704-b36c-2ac1ff69fe8d)

![image](https://github.com/daniel4725/BirdsEyeView/assets/95569050/df1781b1-dc27-4804-a6e2-095713141998)

![image](https://github.com/daniel4725/BirdsEyeView/assets/95569050/f3f512c2-6bbd-4798-ac42-b97d083c63bf)

![image](https://github.com/daniel4725/BirdsEyeView/assets/95569050/d0c5db7f-59ba-496c-a8fc-d71f28eabb19)


## Step C: Stitching

![image](https://github.com/daniel4725/BirdsEyeView/assets/95569050/722a9958-ed0f-4aa3-b0da-2e7f3b5f03de)

## Results

![image](https://github.com/daniel4725/BirdsEyeView/assets/95569050/210924f5-e0a6-4b7e-a4fd-c23451aaf53f)

![image](https://github.com/daniel4725/BirdsEyeView/assets/95569050/2e3b16b7-b26f-46ca-a22d-bd79aa88fd09)

![image](https://github.com/daniel4725/BirdsEyeView/assets/95569050/5355683b-547d-4bc0-bd63-adac8c39b211)


![image](https://github.com/daniel4725/BirdsEyeView/assets/95569050/99fb9801-a595-4727-bc36-02924409ef01)






