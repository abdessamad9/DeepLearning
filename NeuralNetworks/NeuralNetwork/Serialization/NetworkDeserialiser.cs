using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using NeuralNetwork.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Activators;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;

namespace NeuralNetwork.Serialization
{
    public static class NetworkDeserializer
    {
        public static Network Deserialize(SerializedNetwork serializedNetwork)
        {
            Network network = new Network( serializedNetwork.BatchSize, serializedNetwork.SerializedLayers.Length );

            
            // Console.WriteLine("serializedNetwork.BatchSize" +serializedNetwork.BatchSize);
            for (int i = 0 ; i < serializedNetwork.SerializedLayers.Length; i++)
            {
                network.Layers[i] = DeserializeLayer(serializedNetwork.SerializedLayers[i], network.BatchSize, i+1);
                
            }
            network.Output = Matrix<double>.Build.Dense(network.Layers[network.Layers.Length -1].Activation.RowCount , network.Layers[network.Layers.Length -1].Activation.ColumnCount);

   
            return network;
        }


        public static ILayer DeserializeLayer(ISerializedLayer seLayer, int batchSize, double indice)
        {
            IActivator activator;
            ILayer layer;

            switch (seLayer.Type)
            {
                case LayerType.Standard:
                    SerializedStandardLayer standard = (SerializedStandardLayer)seLayer;
                    
                    activator = ActivatorNew(standard.ActivatorType);
                    layer = new Standard(batchSize, activator, standard.Bias, standard.Weights, standard.GradientAdjustmentParameters, indice);

                    return layer;
                case LayerType.L2Penalty:
                    SerializedL2PenaltyLayer l2penalty = (SerializedL2PenaltyLayer)seLayer;
                    SerializedStandardLayer under = (SerializedStandardLayer)(l2penalty.UnderlyingSerializedLayer);
                    activator = ActivatorNew(under.ActivatorType);
                    Standard layerUnder = new Standard(batchSize,activator, under.Bias, under.Weights, under.GradientAdjustmentParameters, indice);
                    activator = ActivatorNew(under.ActivatorType);
                    layer = new L2Penalty(layerUnder, indice, l2penalty.PenaltyCoefficient);
                    return layer;
                /** case LayerType.Dropout:
                     SerializedDropoutLayer dropoutLayer  = (SerializedDropoutLayer)seLayer;
                     layer = new Dropout(dropoutLayer.LayerSize, input, batchSize, activator, biais, activation, weightedError);
                     return layer;
                
                 case LayerType.InputStandardizing:
                     SerializedInputStandardizingLayer inputStandard = (SerializedInputStandardizingLayer)seLayer;
                     layer = new InputStandardizing(inputStandard.LayerSize, input, batchSize, activator, biais, activation, weightedError);
                     return layer;*/
                default:
                    return null;
            }

        }


        public static IActivator ActivatorNew(ActivatorType type)
        {
            switch (type)
            {
                case ActivatorType.Identity:
                    return new Identity();
                case ActivatorType.LeakyReLU:
                    return new LeakyReLU();
                case ActivatorType.ReLU:
                    return new ReLU();
                case ActivatorType.Sigmoid:
                    return new Sigmoid();
                case ActivatorType.Tanh:
                    return new Tanh();
                default:
                    return null;
            }
        }
    }
}
