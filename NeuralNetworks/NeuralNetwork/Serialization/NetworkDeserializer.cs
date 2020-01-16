using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork.Serialization
{
    public static class NetworkDeserializer
    {
        public static Network Deserialize(SerializedNetwork serializedNetwork)
        {
            foreach (ISerializedLayer seLayer in serializedNetwork.SerializedLayers)
            {
               /*switch( seLayer)
                {
                    case SerializedStandardLayer:
                        return new Standatlayer
                }*/
            }
            return null;
        }
    }
}