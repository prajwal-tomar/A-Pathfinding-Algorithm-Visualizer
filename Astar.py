# A* is a type of informed search and here ive implemented it to find the shortest path between a start and an end node
# f_score = g_score + h_score
# where: g_score is the distance between the start node and the current node, h_score is the estimation of teh distance between current and end node
# f_score is the distance between start and the end node if we go through the current node

# Assumptions:
# i'm assuming that the grid is a square
# here ive taken the edge to be of the weight 1 in order to simplify the process

import pygame
import math
# we use PQ in order to get the smallest value out of the PQ in the most efficient way when we implement the algorithm
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)  # Already visited/ closed set
GREEN = (0, 255, 0)  # IN the open set
WHITE = (255, 255, 255)  # Not yet visited
BLACK = (0, 0, 0)  # Obstacle node
PURPLE = (128, 0, 128)  # Path node color
ORANGE = (255, 165, 0)  # Starting node
GREY = (128, 128, 128)  # color of the grid lines
TURQUOISE = (64, 224, 208)  # End color


class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(
            win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        # In this we update the neigbors of the current spot only if they are not a barrier
        self.neighbors = []
        # DOWN, we check if the spot below is valid and is not a barrier
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        # UP, we check if the spot above is valid and is not a barrier
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        # RIGHT, we check if the spot to the right is valid and is not a barrier
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        # LEFT, we check if the spot to the left is valid and is not a barrier
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):  # represents the less than operator
        return False

# This function is used to calculate the heuristic value, using the manhattan distance!


def h(p1, p2):
    x1, x2 = p1
    y1, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


def algorithm(draw, grid, start, end):
    count = 0  # works as a tie breaker in case the fscore of two nodes is the same
    open_set = PriorityQueue()  # keeps the track of node currently in the open set
    open_set.put((0, count, start))  # (f_score, count, node)
    came_from = {}  # determines the parent of the current node
    # distance from the start node to the nth node, it is infinity for all the nodes in the beginning so that it can be updated with the shortest path
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0  # Distance from start node to start node is 0
    # it determines the distance to reach end node from start node through this node
    f_score = {spot: float("inf") for row in grid for spot in row}
    # calls the heuristic function, vaise toh,( f = h + g), but as g is 0 right now hence f becomes equal to h
    f_score[start] = h(start.get_pos(), end.get_pos())

    # helps us see if something is in the open set as it is a set so searching is O(1)
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # we get the element with the minimum f_score
        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:  # if we're at the end node then it means that we are done and hence return true
            # creates the path from the end node to the start node
            reconstruct_path(came_from, end, draw)
            end.make_end()  # in order to not include the end node in the path
            return True

        for neighbor in current.neighbors:
            # adding 1 because we assume that the weight of the edge is 1
            temp_g_score = g_score[current] + 1

            # if this new gscore is less than the current gscore, then update
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + \
                    h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

# This function created the grid using a 2D Matrix


def make_grid(rows, width):
    grid = []
    # gap means the width of the particular spot, acquired by dividing width with no. of rows
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid

# This function draws the line of the grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


# Draws each and every spot on the grid
def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


# retrieves the row and col where the mouse was clicked
def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def main(win, width):
    ROWS = 30
    grid = make_grid(ROWS, width)  # Creates the grid

    start = None
    end = None

    run = True

    while run:  # This is when our program starts running as long as run = True
        # Draws the grid that is all white and with all the grid lines both vertical as well as horizontal
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False  # Stop running the loop

            # Left mouse click is denoted by [0]
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start and spot != end:  # To avoid overwrting the start spot with end spot
                    start = spot
                    start.make_start()

                elif not end and spot != start:
                    end = spot
                    end.make_end()

                elif spot != start and spot != end:
                    spot.make_barrier()

            # Right mouse click is denoted by [2]
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()

                # Doing this because if we reset the start or end spot, then we want to again start with start node and then end node
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                # If the key pressed is space bar and the start and the end node has been determined then only run
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    # here lamba is the anonymous function which can be used to pass a function as a parameter
                    algorithm(lambda: draw(win, grid, ROWS, width),
                              grid, start, end)

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()


main(WIN, WIDTH)
