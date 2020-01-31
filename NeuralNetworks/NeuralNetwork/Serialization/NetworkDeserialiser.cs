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
            for (int i = 0 ; i < serializedNetwork.SerializedLayers.Length; i++)
            {
                network.Layers[i] = DeserializeLayer(serializedNetwork.SerializedLayers[i], network.BatchSize);    
            }
            network.Output = Matrix<double>.Build.Dense(network.Layers[network.Layers.Length -1].Activation.RowCount , network.Layers[network.Layers.Length -1].Activation.ColumnCount);
            return network;
        }

        /**
         * Permet de deserialiser un Layer serialiser
         */
        public static ILayer DeserializeLayer(ISerializedLayer seLayer, int batchSize)
        {
            IActivator activator;
            ILayer layer;

            switch (seLayer.Type)
            {
                case LayerType.Standard:
                    SerializedStandardLayer standard = (SerializedStandardLayer)seLayer;
                    
                    activator = ActivatorNew(standard.ActivatorType);
                    layer = new Standard(batchSize, activator, standard.Bias, standard.Weights, standard.GradientAdjustmentParameters);

                    return layer;
                case LayerType.L2Penalty:
                    SerializedL2PenaltyLayer l2penalty = (SerializedL2PenaltyLayer)seLayer;
                    SerializedStandardLayer under = (SerializedStandardLayer)(l2penalty.UnderlyingSerializedLayer);
                    activator = ActivatorNew(under.ActivatorType);
                    Standard layerUnder = new Standard(batchSize,activator, under.Bias, under.Weights, under.GradientAdjustmentParameters);
                    activator = ActivatorNew(under.ActivatorType);
                    layer = new L2Penalty(layerUnder, l2penalty.PenaltyCoefficient);
                    return layer;
                case LayerType.WeightDecay:
                    SerializedWeightDecayLayer weightDecay = (SerializedWeightDecayLayer)seLayer;
                    SerializedStandardLayer under2 = (SerializedStandardLayer)(weightDecay.UnderlyingSerializedLayer);
                    activator = ActivatorNew(under2.ActivatorType);
                    Standard layerUnder2 = new Standard(batchSize, activator, under2.Bias, under2.Weights, under2.GradientAdjustmentParameters);
                    activator = ActivatorNew(under2.ActivatorType);
                    layer = new WeightDecay(layerUnder2, weightDecay.DecayRate);
                    return layer;
               
                default:
                    return null;
            }

        }


        /**
         * Renvoie la bonne fonction d'activation en fonction du type
         */
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
