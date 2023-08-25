from aitviewer.utils.decorators import hooked
from aitviewer.viewer import Viewer
from cdkg.renderables.garment import Garment
from cdkg.configuration import CONFIG as C

class KGViewer(Viewer):
    """A viewer for editing and displaying kinesthetic garments"""

    title = "Kinesthetic Garments Viewer"

    def __init__(self, **kwargs):
        """
        Initializer.
        :param kwargs: kwargs.`
        """
        super().__init__(config=C._conf, **kwargs)

        # Custom UI
        self._curr_button = None
        self._brush_size = 0
        self._increase_brush_size = self.wnd.keys.RIGHT_BRACKET
        self._decrease_brush_size = self.wnd.keys.LEFT_BRACKET
        self.scene.origin.enabled = False
        self.scene.floor.tiling = False
        self.scene.floor.enabled = False

    @hooked
    def mouse_press_event(self, x: int, y: int, button: int):
        if not self.imgui_user_interacting:
            mmi = self.mesh_mouse_intersection(x, y)
            if mmi is not None:
                self._curr_button = button
                if isinstance(mmi.node, Garment):
                    mmi.node.process_mmi(mmi, self._curr_button, self._right_mouse_button, self._left_mouse_button, self._brush_size)

    @hooked
    def mouse_release_event(self, x: int, y: int, button: int):
        self._curr_button = None

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        self.imgui.mouse_drag_event(x, y, dx, dy)
        is_view = True
        if not self.imgui_user_interacting:
            mmi = self.mesh_mouse_intersection(x, y)
            if mmi is not None:
                if isinstance(mmi.node, Garment):
                    is_view = mmi.node.selected_mode == 'view'
                    mmi.node.process_mmi(mmi, self._curr_button, self._right_mouse_button, self._left_mouse_button, self._brush_size)

        if is_view:
            super().mouse_drag_event(x, y, dx, dy)

    @hooked
    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self._decrease_brush_size:
                self._brush_size -= 1
                self._brush_size = self._brush_size if self._brush_size >= 0 else 0
            elif key == self._increase_brush_size:
                self._brush_size += 1
                self._brush_size = self._brush_size if self._brush_size <= 5 else 5
