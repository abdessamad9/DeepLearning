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
            Layer nlayer = (Layer)layer;
            ILayer under;
            switch (nlayer.Type)
            {
                case LayerType.Dropout:
                     seriaLayer = new SerializedDropoutLayer();
                    return seriaLayer; 

                case LayerType.Standard:
                    Standard layerStandard = (Standard)layer;
                    seriaLayer = new SerializedStandardLayer(layerStandard.Bias.Column(0).ToArray(), layerStandard.Weights.ToArray(), layerStandard.Activator.Type, layerStandard.GradientAdjustmentParameters);
                    return seriaLayer;

                case LayerType.InputStandardizing:
                    seriaLayer = new SerializedInputStandardizingLayer();
                    return seriaLayer;

                case LayerType.L2Penalty:
                    L2Penalty layerPenalty = (L2Penalty)layer;
                    under = layerPenalty.UnderlyingLayer;
                    
                    seriaLayer = new SerializedL2PenaltyLayer( SerializeLayer(under), layerPenalty.PenaltyCoefficient);
                    return seriaLayer;

                case LayerType.WeightDecay:
                    WeightDecay weight = (WeightDecay)layer;
                    //    under = weight.UnderlyingLayer;

                    seriaLayer = new SerializedWeightDecayLayer(); //(SerializeLayer(under), weight.DecayRate);
                    return seriaLayer;

                default:
                    return null;
            }
            

        }

    }

}