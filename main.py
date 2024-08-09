from PuzzleDetector import PuzzleDetector
from Solver import Solver
from DigitClassifier import DigitClassifier

import cv2
import sys
from timeit import default_timer as timer

#TODO Arguments parser con sys


info = {
    'game': 'sudoku', # sudoku, stars, skyscrapers
    'GRID_LEN': 4,
    'SQUARE_LEN': 2,
    'NUM_STARS': 1
}

if len(sys.argv) > 1:#TODO check + controllo games
    if sys.argv[1] is not None:
        info['game'] = sys.argv[1]
    if sys.argv[2] is not None:
        info['GRID_LEN'] = int(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] is not None:
        info['SQUARE_LEN'] = int(sys.argv[3])

puzzle_detected = False
puzzle_analyzed = False
puzzle_solved = False

detector = PuzzleDetector(info)
classifier = DigitClassifier()
solver = Solver(info)

REAL_TIME = False

# 1. Board detection phase

if REAL_TIME:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    counter = 0
    while not puzzle_detected:
        _, frame = cap.read()

        detector.detectGameBoard(frame)
        counter += 1
        if counter % 10 == 0 and detector.grid_digit_images is not None:
            classifier.save_puzzle(detector.grid_digit_images)

        k = cv2.waitKey(20)
        if k == 27 & 0xFF == ord('q'):
            break
        elif k == 32:
            puzzle_detected = True

else:
    img = cv2.imread('input_'+info['game']+'.png')
    detector.detectGameBoard(img)
    classifier.save_puzzle(detector.grid_digit_images)

print('Digits found ', str(len([d for d in detector.grid_digit_images if d is not None])))


# 2. Board analyze phase
start = timer()

digits_found = {}
if info['game'] == 'sudoku':
    digits_found = classifier.get_sudoku_digits(info)
elif info['game'] == 'stars':
    digits_found = detector.get_stars_areas(classifier.puzzles)
elif info['game'] == 'skyscrapers':
    digits_found = classifier.get_skyscrapers_digits(info)

print(digits_found)

# 3. Game solution phase
data = {
    'variables_found': digits_found
}
solved = solver.solveGame(data)

def save_image(image, title, file_prefix="output"):
    """ Save image to a file with a unique name based on the title. """
    if not os.path.exists("images"):
        os.makedirs("images")
    filename = f"imgs/{file_prefix}_{title.replace(' ', '_').replace(':', '')}.png"
    cv2.imwrite(filename, image)
    
# cv2.imshow("Digits found", solver.drawResult(detector.grid_image, digits_found))
end = timer()
print(end - start, " seconds")
if solved != "SOLVED":
    print("Error: ", solved)
    #cv2.imshow("Impossible without solution (wrong ocr?)", solver.drawResult(detector.grid_image, digits_found))
    result_image = solver.drawResult(detector.grid_image, digits_found)
    save_image(result_image, "Impossible_without_solution_wrong_ocr")
else:
    print("Game solved?: ", solved)
    #cv2.imshow("Game Solved in "+ str(end - start)+" seconds", solver.drawResult(detector.grid_image, solver.result))
    result_image = solver.drawResult(detector.grid_image, solver.result)
    save_image(result_image, f"Game_Solved_in_{end - start}_seconds")

#cv2.waitKey(0)

cap.release()
#cv2.destroyAllWindows()
