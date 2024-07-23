import time
import edgeiq
import numpy as np
import cv2
"""
Simultaneously use object detection to detect human faces and classification to classify
the detected faces in terms of age groups, and output results to
shared output stream.

To change the computer vision models, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://docs.alwaysai.co/application_development/application_configuration.html#change-the-engine-and-accelerator
"""
def swap_faces(obox, nbox, image1, image2):
    """[summary]

    Args:
        obox (BoundingBox): The original box
        nbox (BoundingBox): The new box
        image1 (numpy array): The original frame
        image2 (numpy array): The new frame

    Returns:
        numpy array: The new numpy array with the new face on the old face
    """
    face = edgeiq.resize(edgeiq.cutout_image(image1, obox), nbox.width, nbox.height)
    image2[nbox.start_y:nbox.start_y + face.shape[0], nbox.start_x:nbox.start_x + face.shape[1]] = face
    return image2              

def main():
    # first make a detector to detect facial objects
    facial_detector = edgeiq.ObjectDetection(
            "alwaysai/res10_300x300_ssd_iter_140000")
    facial_detector.load(engine=edgeiq.Engine.DNN)

    # descriptions printed to console
    print("Engine: {}".format(facial_detector.engine))
    print("Accelerator: {}\n".format(facial_detector.accelerator))
    print("Model:\n{}\n".format(facial_detector.model_id))

    fps = edgeiq.FPS()

    try:    
        # loop detection
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, edgeiq.Streamer() as streamer:
            time.sleep(2.0)
            fps.start()
            while True:
                frame = video_stream.read()

                # detect human faces
                results = facial_detector.detect_objects(
                        frame, confidence_level=.4)

                # append each predication to the text output
                pred = results.predictions
                frame2 = frame.copy()
                for i in range(0, len(pred)):
                    if len(pred) != 1:   
                        if i < len(pred) - 1:
                            obox = pred[i].box
                            nbox = pred[i + 1].box
                        else:
                            obox = pred[i].box
                            nbox = pred[0].box
                        frame2 = swap_faces(obox, nbox, frame, frame2)

                if len(pred) > 0:
                    # swap the last face with the first face
                    obox = pred[len(pred) - 1].box
                    nbox = pred[0].box
                    frame2 = swap_faces(obox, nbox, frame, frame2)

                # send the image frame and the predictions to the output stream
                frame = edgeiq.markup_image(
                        frame, results.predictions, show_labels=False)
                display_frame = np.concatenate((frame, frame2))
                streamer.send_data(display_frame, "")

                fps.update()
                streamer.wait()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
