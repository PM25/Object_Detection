import cv2


class Image:
    def __init__(self, img_tensor):
        self.img = self.tensor_to_np(img_tensor)
        self.height, self.width, self.channels = self.img.shape


    def tensor_to_np(self, img_tensor):
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        return img


    def draw_rect(self, rect, color=(255, 0, 0)):
        (x, y, w, h) = rect
        x *= self.width
        y *= self.height
        w *= self.width
        h *= self.height
        self.img = cv2.rectangle(self.img, (x - (w/2), y - (h/2)), (x + (w/2), y + (h/2)), color, 3)


    def show(self):
        cv2.imshow("Title", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
