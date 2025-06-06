from config.dataset_paths_selector import dataset_images_path_selector
import re
import glob
import os

'''
logs content into file at log_file_path
if overwrite is Ture, overwrites file if it exists, otherwise just appends 
'''
def log(log_file_path, content, overwrite=False):
    mode = 'w' if overwrite else 'a'
    with open(log_file_path, mode) as log_file:
        log_file.write(content + '\n')
    log_file.close()

'''
checks if labels in file at file_path match labels in class_labels and returns result
    ** assumes each line in file corresponds to label for a seperate instance of a class
    ** labels are formatted with the class label first and the segmentation label second, seperated by a single space

file_path: path to file containing labels
class_labels: correct class label(s) to check against
    * if class_labels is a list, must contain exactly one label for each line in file 
    * if class_labels is not a list, then file must only contain one label
fix: if True and labels in file don't match labels in class_labels, overwrites file with correct class_labels
log_change: if True then function will log any changes made to log_file_path. default True.
log_file_path: path to file where changes will be logged if log_change is True. Ignored if log_change is False. default './log_changes.txt'. 
'''
def labels_match(file_path, class_labels, fix=True, log_change=True, log_file_path='./log_changes.txt'):
    with open(file_path, 'r') as file:
        if isinstance(class_labels, list):
            orig_content = file.readlines()
            data = list(zip(class_labels, orig_content))
            updated_content = list(map(lambda x: str(x[0]) + ' ' + (' ').join(x[1].split(' ')[1:]), data)) 
            orig_content = ('').join(orig_content)
            updated_content = ('').join(updated_content)
        else:
            orig_content = file.read()
            data = orig_content.split(' ')
            updated_content = str(class_labels) + ' ' + (' ').join(data[1:])

    file.close()
    if orig_content == updated_content:
        return True
    elif fix:
        with open(file_path, 'w') as file:
            file.write(updated_content)
        file.close()
        if log_change:
            log_data = (',').join([file_path, orig_content, updated_content])
            log(log_file_path, log_data)
    return False

def main(label_paths, log_file_path, overwrite=False):
    #gets list of all label files
    file_paths = glob.glob(label_paths + "/*.txt", recursive=True)
    #function that gets the pill name given a path to pill file
    getPillName = lambda x: re.match(r'(.+)labels/(.+)_(s|u)_\d+\.txt', x).group(2)
    #converts list of file paths to list of all pill names where index of each pill name corresponds to its' numeric label
    pill_names = sorted(set(map(getPillName, file_paths)))

    if overwrite or not os.path.exists(log_file_path):
        #writes header
        log(log_file_path, "file,original content,updated content", overwrite=True)
    
    for path in file_paths:
        #gets pill name from path and gets index of that pill name
        name = getPillName(path)
        idx = str(pill_names.index(name))
        
        #checks if labels match and if not, fixes them
        labels_match(path, idx, fix=True, log_change=True, log_file_path=log_file_path)


if __name__ == "__main__":
    log_file_path = "./log_changes.txt"
    unplitted_labels = dataset_images_path_selector("ogyeiv2").get("unsplitted").get("segmentation_labels")

    main(unplitted_labels, log_file_path)

    customer_log_path = "./log_changes.txt"
    customer_labels = dataset_images_path_selector("ogyeiv2").get("customer").get("customer_segmentation_labels")
    main(customer_labels, log_file_path)

    reference_log_path = "./log_changes.txt"
    reference_labels = dataset_images_path_selector("ogyeiv2").get("reference").get("reference_segmentation_labels")
    main(reference_labels, log_file_path)
