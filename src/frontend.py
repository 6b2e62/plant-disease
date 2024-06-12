import os

import gradio as gr
import matplotlib as mpl
import numpy as np
import tensorflow as tf

import models as mod
from dataset.consts import ALL_CLASSES
from trainer.trainer import Trainer
from src.transfer_learning import load_args, load_model

os.environ["WANDB_MODE"] = "offline"

args = load_args()
model = load_model(args)
model.load_weights(f"weights/transfer_learning/resnet50v2_256.keras")

preprocess_input = Trainer.choose_preprocess_fn(model)
model_class = model.__class__


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(
        model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    return superimposed_img


def predict(img):
    img = img.resize((int(args.size), int(args.size)))
    img = np.array(img)
    img_orig = img.copy()

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predictions = model.predict(img)

    if model_class == mod.mobilenetv2.MobilenetV2Model:
        heatmap = make_gradcam_heatmap(img, model.model, "out_relu", [
                                       "global_average_pooling2d", "dropout", "dense"])
    elif model_class == mod.resnet50v2.Resnet50V2Model:
        heatmap = make_gradcam_heatmap(img, model.model, "post_relu", [
                                       "global_average_pooling2d", "dense"])
    elif model_class == mod.efficentnetv2b0.EfficientNetV2B0Model:
        heatmap = make_gradcam_heatmap(img, model.model, "top_activation", [
                                       "global_average_pooling2d", "dropout", "dense"])

    img_heatmap = save_and_display_gradcam(img_orig, heatmap)

    return ALL_CLASSES[np.argmax(predictions[0], axis=-1)], img_heatmap


demo = gr.Interface(
    predict,
    gr.Image(type="pil", width=800, height=600),
    ["text", gr.Image()],
)

demo.launch()
