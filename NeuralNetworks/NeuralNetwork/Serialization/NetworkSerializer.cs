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
            foreach( ILayer layer in network.Layers)
            {
               return SerializeLayer(layer);
            }
            return null;
        }

        public static SerializedNetwork SerializeLayer(ILayer layer)
        {
           /* switch (layer)
            {
                case SerializedInputStandardizingLayer:
                    return ....
            }*/
            return null;
        }


    }
}