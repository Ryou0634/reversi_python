from typing import List, Optional, NamedTuple, Callable
import tkinter
from tkinter import Button, Canvas, StringVar, Label
import tkinter.messagebox

from reversi.board import Color, Position, ReversiBoard
from .game_interface import GameInterface


class Disk(NamedTuple):
    obj: object
    color: Color


class BoardGrids:
    disk_size_ratio: float = 0.8
    color_name_dict = {Color.WHITE: "white", Color.BLACK: "black"}

    def __init__(self, app, board_size: int, canvas_size: int = 400, board_color: str = "lime green"):
        self.board_color = board_color
        self.canvas = Canvas(app, bg=board_color, width=canvas_size, height=canvas_size, highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)

        assert canvas_size % (board_size + 2) == 0
        cell_size = canvas_size // (board_size + 2)

        grid_points: List[int] = list(range(cell_size, canvas_size, cell_size))

        self.rectangles: List[List[str]] = [[None for _ in range(board_size)] for _ in range(board_size)]
        for x in range(board_size):
            for y in range(board_size):
                tag = f"position_{x}{y}"
                self.canvas.create_rectangle(
                    *(grid_points[i] for i in [y, x, y + 1, x + 1]), fill=self.board_color, tags=tag
                )
                self.rectangles[x][y] = tag

        self.cells: List[List[Optional[Disk]]] = [[None] * board_size for i in range(board_size)]

    def bind_rectangles(self, bind_func: Callable):
        for recs in self.rectangles:
            for rectangle in recs:
                self.canvas.tag_bind(rectangle, "<ButtonPress-1>", bind_func)

    def _draw_disk(self, x: int, y: int, color: Optional[Color]):

        if (self.cells[x][y] is None and color is None) or (
            self.cells[x][y] is not None and self.cells[x][y].color == color
        ):
            return

        if self.cells[x][y] is not None:
            self.canvas.delete(self.cells[x][y].obj)
            self.cells[x][y] = None

        if color is None:
            return

        x_top, y_top, x_bottom, y_bottom = self.canvas.coords(self.rectangles[x][y])
        center_x = (x_top + x_bottom) // 2
        center_y = (y_top + y_bottom) // 2
        disk_radius = ((x_bottom - x_top) * self.disk_size_ratio) // 2

        xs = center_x - disk_radius
        ys = center_y - disk_radius
        xe = center_x + disk_radius
        ye = center_y + disk_radius
        disk = self.canvas.create_oval(xs, ys, xe, ye, fill=self.color_name_dict[color])
        self.cells[x][y] = Disk(disk, color)

    def update_with_board(self, board: ReversiBoard, current_color: Color):
        legal_positions = set(board.get_legal_positions(color=current_color))

        for x in range(board.size):
            for y in range(board.size):
                position = Position(x, y)
                if position in legal_positions:
                    self.canvas.itemconfig(self.rectangles[x][y], fill="green")
                else:
                    self.canvas.itemconfig(self.rectangles[x][y], fill=self.board_color)

                color = board.get_color(position)
                self._draw_disk(x, y, color)

    def get_closest_position(self, x_pixel: int, y_pixel: int) -> Position:
        id = self.canvas.find_closest(x_pixel, y_pixel)
        tag = self.canvas.gettags(id[0])[0]
        x, y = map(int, tag.split("_")[1])
        return Position(x, y)


@GameInterface.register("tklinter")
class TklinterGUI(GameInterface):
    def __init__(self, canvas_size: int = 400, board_color: str = "lime green", **kwargs):
        super().__init__(**kwargs)
        self.app = tkinter.Tk()
        self.app.title("Reversi")

        self.canvas_size = canvas_size
        self.board_color = board_color

        self.grids = BoardGrids(
            self.app, board_size=self.game_engine.board.size, canvas_size=canvas_size, board_color=board_color
        )
        self.grids.bind_rectangles(self.place_disk_by_human)

        self._init_board()

        self.forward_button = Button(self.app, text="➡︎", command=self.place_disk_by_computer)
        self.forward_button.pack()

        self.backward_button = Button(self.app, text="⬅︎", command=self.backward)
        self.backward_button.pack()

    def _init_board(self):

        # set label
        self.info = Canvas(self.app, bg="white", width=300, height=100)
        self.info.pack()
        self.var = StringVar()
        self.label = Label(self.info, bg="white", font=("Helvetica", 15), textvariable=self.var)
        self.label.pack()

        self.game_engine.reset()
        self._update_board()

    def _update_board(self):
        self.grids.update_with_board(self.game_engine.board, self.game_engine.current_color)
        self.var.set(f"{self.game_engine.current_color}'s turn")

    def place_disk_by_human(self, event):
        position = self.grids.get_closest_position(event.x, event.y)
        legal_positions = self.game_engine.board.get_legal_positions(self.game_engine.current_color)
        if position not in legal_positions:
            return

        self._place_disk(position)

    def place_disk_by_computer(self):
        legal_positions = self.game_engine.board.get_legal_positions(self.game_engine.current_color)
        position = self.players[self.game_engine.current_color].choose_position(self.game_engine.board, legal_positions)
        self._place_disk(position)

    def _place_disk(self, position: Position):
        is_terminal = self.game_engine.execute_move(position)
        self._update_board()
        if is_terminal:
            result = self.game_engine.summarize_result()
            tkinter.messagebox.showinfo(title="Result", message=result.message)

    def backward(self):
        self.game_engine.restore_from_latest_snapshot()
        self._update_board()

    def play(self):
        self.app.mainloop()
