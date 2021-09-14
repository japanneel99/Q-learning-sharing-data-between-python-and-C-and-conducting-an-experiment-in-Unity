using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Feeling : MonoBehaviour
{

    public float did_You_Feel_Safe;
    Vector3 wheelchairPosition;
    Vector3 pedestrianPosition;


    [SerializeField]
    public GameObject Pedestrian;


    // Start is called before the first frame update
    void Start()
    {
        did_You_Feel_Safe = 0.0f;
        wheelchairPosition = transform.position;
        pedestrianPosition = Pedestrian.transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        wheelchairPosition = transform.position;
        pedestrianPosition = Pedestrian.transform.position;

        float wheelchair_x_position = transform.position.x;
        float wheelchair_z_position = transform.position.z;
        float pedestrian_x_position = Pedestrian.transform.position.x;
        float pedestrian_z_position = Pedestrian.transform.position.z;

        float relative_x_distance = pedestrian_x_position - wheelchair_x_position;
        float relative_z_distance = pedestrian_z_position - wheelchair_z_position;

        float inside_relative = Mathf.Pow(relative_x_distance, 2) + Mathf.Pow(relative_z_distance, 2);
        float relative_distance = Mathf.Sqrt(inside_relative);

        if (Input.GetKeyDown(KeyCode.Space))
        {
            did_You_Feel_Safe = 1.0f;
            Debug.Log("The Space bar was pressed ");
        }
        else if (relative_distance < 0.4)
        {
            did_You_Feel_Safe = -10.0f;
        }
        else
        {
            did_You_Feel_Safe = 0.0f;
        }
    }
}