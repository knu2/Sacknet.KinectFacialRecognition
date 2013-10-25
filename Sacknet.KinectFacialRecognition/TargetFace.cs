using Microsoft.Kinect.Toolkit.FaceTracking;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sacknet.KinectFacialRecognition
{
    /// <summary>
    /// Describes a target face for facial recognition
    /// </summary>
    public class TargetFace
    {
        /// <summary>
        /// Gets or sets the key returned when this face is found
        /// </summary>
        public string Key { get; set; }

        /// <summary>
        /// Gets or sets the grayscale, 100x100 target image
        /// </summary>
        public Bitmap Image { get; set; }

        /// <summary>
        /// Gets or sets the face 3D points
        /// </summary>
        public EnumIndexableCollection<FeaturePoint, Vector3DF> Face3DPoints { get; set; }

        /// <summary>
        /// Gets or sets the unique ID when this face is found
        /// </summary>
        public int ID { get; set; }
    }

    public static class RecognitionUtility
    {
        public static int GenerateHash(string s)
        {
            int h = 0;
            for (int i = 0; i < s.Length; i++)
            {
                h = 31 * h + s[i];
            }
            return h;
        }

        /// <summary>
        /// Removes the least significant 2 digits. Used for Hash generation and name lookup
        /// </summary>
        /// <param name="?">id to shorten</param>
        /// <returns>shortened id</returns>
        public static int Shorten(int id)
        {
            return id / 100;
        }

    }
}
