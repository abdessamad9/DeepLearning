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
            Network network = new Network();
            network.BatchSize = serializedNetwork.BatchSize;
            network.Layers = new ILayer[network.BatchSize];
            int indice = 0;
            foreach (ISerializedLayer seLayer in serializedNetwork.SerializedLayers)
            {
                network.Layers[indice] = DeserializeLayer(seLayer, network.BatchSize);
            }
            return network;
        }


        public static ILayer DeserializeLayer(ISerializedLayer seLayer, int batchSize)
        {

            Matrix<double> biais;
            Matrix<double> weigths;
            //Matrix<double> activation = Matrix<double>.Build.Dense(batchSize, batchSize);
            //La matrice activation
            Matrix<double> weightedError = Matrix<double>.Build.Dense(batchSize, batchSize);
            IActivator activator;
            int input = 0;

            ILayer layer;
            switch (seLayer.Type)
            {
                case LayerType.Standard:
                    SerializedStandardLayer standard = (SerializedStandardLayer)seLayer;
                    activator = ActivatorNew(standard.ActivatorType);
                    activator = ActivatorNew(standard.ActivatorType);
                    biais = new DenseMatrix(standard.Bias.Length, standard.Bias.Length, standard.Bias);
                    weigths = DenseMatrix.OfArray(standard.Weights);
                    layer = new Standard(0, input, batchSize, activator, biais, weigths, weightedError);

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
