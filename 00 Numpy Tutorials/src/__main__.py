from pattern import *
from generator import *



def main():

    
    print('Select any one:  1.checker,  2.circle,  3.spectrum, 4. Image generator')
    draw = input('What do you want to draw?:')
        

    if(draw == '1'):
        ## 1. Checker board
        print('#'*40)
        print('Building Checker-Board')
        print('#'*40)
        tile_size = int(input('Enter tile size for checker board:'))#1
        resolution = int(input('Enter resolution for checker board:'))#10
        if resolution % (tile_size*2) != 0:
            print('Invalid size')
        else:
            checker_board = Checker(resolution, tile_size)
            print('Here is your checker-board...')
            print('#'*40)
            checker_board.show()

    elif(draw == '2'):

        ## 2. Circle
        print('#'*40)
        print('Building Circle')
        print('#'*40)
        radius = int(input('Enter radius:'))
        resolution = int(input('Enter resolution:'))
        position = input('Enter position:')
        position = tuple(map(int, position.split(',')))
        
        c = Circle(resolution, radius, position)
        print('Here is your circle...')
        print('#'*40)
        c.show()

    elif(draw == '3'):

        ## 3. Spectrum
        print('#'*40)
        print('Building Spectrum')
        print('#'*40)
        resolution = int(input('Enter resolution:'))
        
        s = Spectrum(resolution)
        print('Here is your spectrum...')
        print('#'*40)
        s.show()

    elif (draw == '4'):
        x = ImageGenerator('src_final/exercise_data','src_final/Labels.json',12,(32,32,3),True, True, True)
        x.show()

    else:
        print('Invalid choice!')

    


if __name__ == '__main__':
    main()