using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;

public class ExampleConnection : MonoBehaviour
{
    Thread mThread;
    public string connectionIP = "127.0.0.1";
    public int connectionPort = 25001;
    TcpListener listener;
    TcpClient client;
    Vector3 velocity;
    Vector3 position;
    IPAddress localAdd;
    bool running;

    private void Update()
    {
        position = transform.position;
        velocity = GetComponent<Rigidbody>().velocity;
    }

    private void Start()
    {
        GetComponent<Rigidbody>().velocity = new Vector3(1.0f, 0.0f, 0.0f);
        position = transform.position;
        velocity = GetComponent<Rigidbody>().velocity;

        ThreadStart ts = new ThreadStart(GetInfo);
        mThread = new Thread(ts);
        mThread.Start();
    }

    void GetInfo()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(IPAddress.Any, connectionPort);
        listener.Start();

        client = listener.AcceptTcpClient();

        running = true;
        while (running)
        {
            SendAndReceiveData();
        }
        listener.Stop();
    }

    void SendAndReceiveData()
    {
        NetworkStream nwStream = client.GetStream();
        byte[] buffer = new byte[client.ReceiveBufferSize];

        //---receiving Data from the Host----
        int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize); //Getting data in Bytes from Python
        string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead); //Converting byte data to string

        if (dataReceived != null)
        {
            Debug.Log(string.Format("recieved data:{0}", dataReceived));
            var xs = dataReceived.Split(',');
            var cmd = xs[0];
            var send_message = "";

            switch (cmd)
            {
                case "GET_POSITION":
                    send_message = JsonUtility.ToJson(position);
                    break;
                case "GET_VELOCITY":
                    send_message = JsonUtility.ToJson(velocity);
                    break;
                case "SEND_ACTION":
                    send_message = "received action correctly";
                    Debug.Log(string.Format("parsed data: {0}", int.Parse(xs[1])));
                    break;
                default:
                    send_message = "unkown command";
                    break;
            }

            //---Sending Data to Host----
            //Converting string to byte data
            byte[] myWriteBuffer = Encoding.ASCII.GetBytes(send_message);
            nwStream.Write(myWriteBuffer, 0, myWriteBuffer.Length); //Sending the data in Bytes to Python
        }
    }

    public static Vector3 StringToVector3(string sVector)
    {
        // Remove the parentheses
        if (sVector.StartsWith("(") && sVector.EndsWith(")"))
        {
            sVector = sVector.Substring(1, sVector.Length - 2);
        }

        // split the items
        string[] sArray = sVector.Split(',');

        // store as a Vector3
        Vector3 result = new Vector3(
            float.Parse(sArray[0]),
            float.Parse(sArray[1]),
            float.Parse(sArray[2]));

        return result;
    }
    /*
    public static string GetLocalIPAddress()
    {
        var host = Dns.GetHostEntry(Dns.GetHostName());
        foreach (var ip in host.AddressList)
        {
            if (ip.AddressFamily == AddressFamily.InterNetwork)
            {
                return ip.ToString();
            }
        }
        throw new System.Exception("No network adapters with an IPv4 address in the system!");
    }
    */
}