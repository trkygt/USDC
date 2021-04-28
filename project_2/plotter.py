import matplotlib.pyplot as plt
import cv2


def plot_images(images, titles, shape, cmap, figname):
    """
    This function is used to plot subplots in a more compact manner in the actual pipeline and main.py.
    """
    rows = shape[0]
    cols = shape[1]
    if rows == 1 and cols == 1:
        fig = plt.figure(figsize=(24, 9))
        fig.tight_layout()
        if cmap == 1:
            plt.imshow(images, cmap='gray')
        else:
            plt.imshow(images)
        plt.title(titles, size=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    else:
        fig, axs = plt.subplots(rows, cols, figsize=(24, 9))
        fig.tight_layout()
        if rows == 1 or cols == 1:
            for i in range(cols):
                if cmap[i] == 1:
                    axs[i].imshow(images[i], cmap='gray')
                else:
                    axs[i].imshow(images[i])
                axs[i].set_title(titles[i], size=20)
        else:
            for i in range(1, rows+1):
                for j in range(1, cols+1):
                    c = (j-1) * rows + i-1
                    if cmap[c] == 1:
                        axs[i - 1, j - 1].imshow(images[c], cmap='gray')
                    else:
                        axs[i - 1, j - 1].imshow(images[c])
                    axs[i-1, j-1].set_title(titles[c], size=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    fig.savefig("output_images/" + figname + ".jpg")
    plt.show()


def plot_trapezoid(img, top_view):
    """
    Plot sanity check for perspective transform
    """
    # Define trapezoid vertices
    vertices = [(556, 480), (734, 480), (220, 720), (1105, 720)]

    img = cv2.line(img, vertices[0], vertices[1], (255, 0, 0), thickness=8)
    img = cv2.line(img, vertices[1], vertices[3], (255, 0, 0), thickness=8)
    img = cv2.line(img, vertices[3], vertices[2], (255, 0, 0), thickness=8)
    img = cv2.line(img, vertices[2], vertices[0], (255, 0, 0), thickness=8)

    # Rectangle Vertices
    corners = [(300, 0), (900, 0), (300, 720), (900, 720)]
    top_view = cv2.line(top_view, corners[0], corners[1], (255, 0, 0), thickness=8)
    top_view = cv2.line(top_view, corners[1], corners[3], (255, 0, 0), thickness=8)
    top_view = cv2.line(top_view, corners[3], corners[2], (255, 0, 0), thickness=8)
    top_view = cv2.line(top_view, corners[2], corners[0], (255, 0, 0), thickness=8)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    fig.tight_layout()
    ax1.imshow(img)
    ax1.set_title("Lanes on Undistorted Image", size=20)
    ax2.imshow(top_view)
    ax2.set_title("Lanes on Bird's View Image", size=20)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    fig.savefig("output_images/" + "Sanity_Check_for_PT.jpg")
    plt.show()



