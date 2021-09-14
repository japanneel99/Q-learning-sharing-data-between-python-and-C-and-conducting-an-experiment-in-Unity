using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;

public class Connection : MonoBehaviour
{
    [SerializeField]
    public GameObject Pedestrian;

    Thread mThread;
    public string connectionIP = "127.0.0.1";
    public int connectionPort = 25001;
    TcpListener listener;
    TcpClient client;
    Vector3 wheelchairVelocity;
    Vector3 wheelchairPosition;
    Vector3 pedestrianPosition;
    Vector3 pedestrianVelocity;
    public float feeling;
    IPAddress localAdd;
    bool running;


    // Update is called once per frame
    private void Update()
    {
        wheelchairPosition = transform.position;
        wheelchairVelocity = this.GetComponent<PMController>().CurrentVelocity;

        pedestrianPosition = Pedestrian.transform.position;
        pedestrianVelocity = Pedestrian.GetComponent<NetworkCharacterVelocity>().velocity;

        feeling = GetComponent<Feeling>().did_You_Feel_Safe;


    }

    private void Start()
    {
        wheelchairPosition = transform.position;
        wheelchairVelocity = this.GetComponent<PMController>().CurrentVelocity;

        pedestrianPosition = Pedestrian.transform.position;
        pedestrianVelocity = Pedestrian.GetComponent<NetworkCharacterVelocity>().velocity;

        feeling = GetComponent<Feeling>().did_You_Feel_Safe;

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
        while(running)
        {
            SendAndReceiveData();
        }
        listener.Stop();
    }

    void SendAndReceiveData()
    {
        NetworkStream nwStream = client.GetStream();
        byte[] buffer = new byte[client.ReceiveBufferSize];

        //----receiving data from host-----//
        int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize); //Getting data in bytes from python. パイソンからバイトでデータを取得
        string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead); // Converting byte data to string. バイトデータをストリングに変換

        if (dataReceived != null)
        {
            Debug.Log(string.Format("received data:{0}", dataReceived));
            var xs = dataReceived.Split(',');
            var cmd = xs[0];
            var send_message = "";

            switch (cmd)
            {
                case "GET_WHEELCHAIR_POSITION":
                    send_message = JsonUtility.ToJson(wheelchairPosition);
                    break;
                case "GET_WHEELCHAIR_VELOCITY":
                    send_message = JsonUtility.ToJson(wheelchairVelocity);
                    break;
                case "GET_PEDESTRIAN_POSITION":
                    send_message = JsonUtility.ToJson(pedestrianPosition);
                    break;
                case "GET_PEDESTRIAN_VELOCITY":
                    send_message = JsonUtility.ToJson(pedestrianVelocity);
                    break;
                case "GET_FEELING_REWARD":
                    send_message = JsonUtility.ToJson(feeling);
                    break;
                case "SEND_ACTION":
                    send_message = "received action without any issues";
                    Debug.Log(string.Format("parsed data: {0}", int.Parse(xs[1])));
                    break;
                default:
                    send_message = "unknown command";
                    break;
            }
            //---Sending data to python... パイソンにデータを送信
            //---Converting string to byte data ストリングをバイトに変換
            byte[] myWriteBuffer = Encoding.ASCII.GetBytes(send_message);
            nwStream.Write(myWriteBuffer, 0, myWriteBuffer.Length); //Sending to python.パイソンに送信
        }

    }

    public static Vector3 StringToVector3(string sVector)
    {
        //remove the parentheses
        if (sVector.StartsWith("(") && sVector.EndsWith(")"))
        {
            sVector = sVector.Substring(1, sVector.Length - 2);
        }

        //split the items, 要素を分解する
        string[] sArray = sVector.Split(',');

        //store as a vector3 Vector3として保存する
        Vector3 result = new Vector3(
            float.Parse(sArray[0]),
            float.Parse(sArray[1]),
            float.Parse(sArray[2]));

        return result;
    }

}