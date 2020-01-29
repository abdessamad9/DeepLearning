using NeuralNetwork.Common;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Layers;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;

namespace NeuralNetwork.Serialization
{
    public static class NetworkSerializer
    {        

        public static SerializedNetwork Serialize(INetwork network)
        {
            SerializedNetwork serializedNetwork = new SerializedNetwork();
            serializedNetwork.BatchSize = network.BatchSize;
            serializedNetwork.SerializedLayers = new ISerializedLayer[network.Layers.Length];
         
            for (int i = 0; i < network.Layers.Length; i++)
            {
                serializedNetwork.SerializedLayers[i] = SerializeLayer(network.Layers[i]);
            }

          
            return serializedNetwork;
        }

        public static ISerializedLayer SerializeLayer(ILayer layer)
        {
            ISerializedLayer seriaLayer;
            /*switch (layer.Type)
            {
                case LayerType.Dropout:
                     seriaLayer = new SerializedDropoutLayer();
                    return seriaLayer; 

                case LayerType.Standard:
                    seriaLayer = new SerializedStandardLayer();
                    return seriaLayer;
             
                case LayerType.InputStandardizing:
                    seriaLayer = new SerializedInputStandardizingLayer();
                    return seriaLayer;

                case LayerType.L2Penalty:
                    seriaLayer = new SerializedL2PenaltyLayer();
                    return seriaLayer;

                case LayerType.WeightDecay:
                    seriaLayer = new SerializedWeightDecayLayer();
                    return seriaLayer;

                default:
                    return null;
            }*/
            Standard layerStandard = (Standard)layer;
            seriaLayer = new SerializedStandardLayer(layerStandard.Bias.Column(0).ToArray(), layerStandard.Weights.ToArray(),layerStandard.Activator.Type, layerStandard.GradientAdjustmentParameters);
            return seriaLayer;

        }

    }

}