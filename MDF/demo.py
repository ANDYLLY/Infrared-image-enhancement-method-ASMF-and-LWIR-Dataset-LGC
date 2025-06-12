from MDF import MDF_Net

def main():
    # Initialize your model and other components here
    model = MDF_Net(nn_img_num=16)
    model.train_test()

if __name__ == '__main__':
    main()