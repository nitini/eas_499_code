import os
import sys
import subprocess

def gen_resize_command(raw_img, size, raw_img_resized):
    dims = ' -resize ' + size + 'x' + size + '\! '
    return 'convert ' + raw_img + dims + raw_img_resized

def gen_rotation_command(raw_img, degree, raw_img_rotated):
    return 'convert ' + raw_img + ' -rotate ' + degree +\
            ' ' + raw_img_rotated

def gen_mirror_command(raw_img, mirror, raw_img_mirrored):
    return 'convert ' + raw_img + ' -' + mirror + ' ' + raw_img_mirrored

def gen_class_map(plankton_classes):
    class_map = {}
    for plankton_class, class_val in zip(plankton_classes, range(len(plankton_classes))):
            class_map[plankton_class] = str(class_val)
    return class_map

def write_train_txt(file_name, class_label, txt_files):
    for txt in txt_files:
        f = open(txt, 'a')
        f.write(file_name + ' ' + class_label + '\n')
        f.close()
        
def main():
    root_original_folder = './train/'
    resize_output_folder = './augmented_resize_train'
    rotation_output_folder = './augmented_rotation_train'
    full_output_folder = './augmented_full_train'

    plankton_class_map = gen_class_map(os.listdir(root_original_folder))

    for size in ['48', '72', '96']:

        print 'Working on Size: ' + size
        resize_output_folder_s = resize_output_folder + '_' + size
        rotation_output_folder_s = rotation_output_folder + '_' + size
        full_output_folder_s = full_output_folder + '_' + size
        os.mkdir(resize_output_folder_s)
        os.mkdir(full_output_folder_s)
        os.mkdir(rotation_output_folder_s)
        plankton_classes = os.listdir(root_original_folder)
        for plankton_class, count in zip(plankton_classes, range(len(plankton_classes))):
            print 'Working on Class ' + str(count) + ': ' + plankton_class
            class_label = plankton_class_map[plankton_class]
            resize_output_folder_s_p = resize_output_folder_s + '/' + plankton_class
            rotation_output_folder_s_p = rotation_output_folder_s + '/' + plankton_class
            full_output_folder_s_p = full_output_folder_s + '/' + plankton_class
            os.mkdir(resize_output_folder_s_p)
            os.mkdir(rotation_output_folder_s_p)
            os.mkdir(full_output_folder_s_p)

            for raw_img in os.listdir(root_original_folder + plankton_class + '/'):
                raw_img_name = raw_img.split('.')[0]
                raw_img_resize_name = class_label + '_' + raw_img_name + '_' + size + '_0.jpg'
                raw_img_orig = root_original_folder + plankton_class + '/' + raw_img
                raw_img_resized = resize_output_folder_s_p + '/' + raw_img_resize_name
                resize_cmd = gen_resize_command(raw_img_orig, size, raw_img_resized)
                subprocess.call(resize_cmd, shell=True)
                copy_resize_cmd_rotated = 'cp ' + raw_img_resized + ' ' + \
                        rotation_output_folder_s_p + '/' + raw_img_resize_name
                copy_resize_cmd_full = 'cp ' + raw_img_resized + ' ' + \
                        full_output_folder_s_p + '/' + raw_img_resize_name
                subprocess.call(copy_resize_cmd_rotated, shell=True)
                subprocess.call(copy_resize_cmd_full, shell=True)

                resize_txt_files = ['augmented_resize_train' + '_' + size + '.txt',
                             'augmented_rotate_train' + '_' + size + '.txt',
                             'augmented_full_train' + '_' + size + '.txt']
                write_train_txt(raw_img_resize_name, plankton_class_map[plankton_class], \
                        resize_txt_files)

                for deg in ['90', '180', '270']:
                    raw_img_rotate_name = class_label + '_' + raw_img_name + '_' + size + '_'\
                            + deg + '.jpg'
                    raw_img_rotated = rotation_output_folder_s_p + '/' + raw_img_rotate_name
                    rotation_cmd = gen_rotation_command(raw_img_resized, deg,\
                            raw_img_rotated)
                    subprocess.call(rotation_cmd, shell=True)
                    copy_rotation_cmd = 'cp ' + raw_img_rotated + ' ' + \
                            full_output_folder_s_p + '/' + raw_img_rotate_name
                    subprocess.call(copy_rotation_cmd, shell=True)

                    rotate_txt_files = ['augmented_rotate_train' + '_' + size + '.txt',
                                        'augmented_full_train' + '_' + size + '.txt']
                    write_train_txt(raw_img_rotate_name, plankton_class_map[plankton_class], \
                            rotate_txt_files)

                for mirror in ['flip', 'flop']:
                    raw_img_mirror_name = class_label + '_' + raw_img_name + '_' + size + \
                            '_' + mirror + '.jpg'
                    raw_img_mirrored = full_output_folder_s_p +  '/' + raw_img_mirror_name
                    mirror_cmd = gen_mirror_command(raw_img_resized, mirror, \
                            raw_img_mirrored)
                    subprocess.call(mirror_cmd, shell=True)

                    write_train_txt(raw_img_mirror_name, plankton_class_map[plankton_class], \
                            ['augmented_full_train' + '_' + size + '.txt'])

if __name__ == "__main__":
    main()
