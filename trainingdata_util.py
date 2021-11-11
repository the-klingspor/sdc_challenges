import numpy as np
from demonstrations import load_demonstrations
import matplotlib.pyplot as plt
import argparse
import os
import re

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()

    main_parser.add_argument(
        "--inspect",
        action="store_true",
        help="inspect a training data set, close a plot to open the next one. Actions are printed in the terminal"
    )
    main_parser.add_argument(
        "--merge",
        action="store_true",
        help="merges 'path'/teacher with all datasets in 'path'/teacher_new"
    )
    main_parser.add_argument(
        "--inspect_path",
        action="store",
        type=str,
        default="data/teacher",
        help="specify the path to the folder with the training data set to be inspected"
    )
    main_parser.add_argument(
        "--merge_path",
        action="store",
        type=str,
        default="data",
        
    )
    main_parser.add_argument(
        "--merge_Folder",
        action="store",
        type=str,
        default="data/teacher_merged",
        help="specify the path where the teacher and teacher_new directories are"
    )

    args = main_parser.parse_args()
    if args.inspect:
        [observations, actions] = load_demonstrations(args.inspect_path)
        if( len(observations) == 0):
            print("no data in this folder")
        for index, image in enumerate(observations):
            print(actions[index])
            plt.figure()
            plt.imshow(image) 
            plt.show()
    elif args.merge:
        # path error handling
        if(args.merge_path == "data" and not os.path.isdir("data")):
            print("default path 'data' was chosen but no directory data was found")
            exit()
        teacher_folder = os.path.join(args.merge_path,"teacher")
        teacher_new_folder = os.path.join(args.merge_path,"teacher_new")
        if not os.path.isdir(teacher_folder):
            print("no teacher folder found in specified path")
            exit()
        if not os.path.isdir(teacher_new_folder):
            print("no teacher_new folder found in specified path")
            exit()
        os.makedirs(args.merge_Folder, exist_ok=True) 

        # merging

        #check windows - linux
        if(os.name == 'nt'):
            cp = 'copy '
            print("please copy the content of teaching over to teaching_merged manually :D")
        else:
            cp = 'cp '
            os.popen('cp -a ' + teacher_folder + '/. ' + args.merge_Folder)
        
        count = len([f for f in os.listdir(teacher_folder) if f.startswith("observation")])

        for dir_number in os.listdir(teacher_new_folder):
            highes_number = 0
            for new_teaching in os.listdir(os.path.join(teacher_new_folder,dir_number)):
                number = int(re.search('\d+',new_teaching).group())
                if number > highes_number: highes_number = number
                source = os.path.join(teacher_new_folder,dir_number,new_teaching)
                destination = os.path.join(args.merge_Folder,re.sub('\d+',str(count + number),new_teaching))
                os.popen(cp + source + ' ' + destination) 
            count +=  highes_number + 1
        

    else:
        print("\nspecify\n    --inspect : to inspect a training data set, close a plot to open the next one. Actions are printed in the terminal\n    --merge': to merge /data/teach with all datasets in /data/teach_new \n    --inspect_path\n    --merge_path"  )



