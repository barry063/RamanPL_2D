import matplotlib.pyplot as plt
from renishawWiRE import WDFReader


class MappingImage:
    """Displays optical images from .wdf files with mapping region overlay."""

    def __init__(self, filename):
        if not filename.endswith(".wdf"):
            raise ValueError("MappingImage can only be used with .wdf files.")
        self.reader = WDFReader(filename)

    def show_optical_image(self):
        from PIL import Image
        import matplotlib.patches as patches

        image = Image.open(self.reader.img)
        cb = self.reader.img_cropbox
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = patches.Rectangle(
            (cb[0], cb[1]),
            cb[2] - cb[0],
            cb[3] - cb[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.title("Optical Image with Mapping Area")
        plt.show()