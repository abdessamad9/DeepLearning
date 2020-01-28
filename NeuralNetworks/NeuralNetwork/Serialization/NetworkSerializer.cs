using NeuralNetwork.Common;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork.Serialization
{
    public static class NetworkSerializer
    {        

        public static SerializedNetwork Serialize(INetwork network)
        {
            SerializedNetwork ns = new SerializedNetwork();
           ns.BatchSize =4;// network.BatchSize;
           ns.SerializedLayers = new ISerializedLayer[network.Layers.Length];
         
            for (int i = 0; i < network.Layers.Length; i++)
            {
                ns.SerializedLayers[i] = SerializeLayer(network.Layers[i]);
            }

          
            return ns;
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
            seriaLayer = new SerializedStandardLayer();
            return seriaLayer;

        }

    }
}