/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2023 UNIVERSIDADE FEDERAL DE GOIÁS
 * Copyright (c) NumbERS - INSTITUTO FEDERAL DE GOIÁS - CAMPUS INHUMAS
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Rogério S. Silva <rogerio.sousa@ifg.edu.br>
 */
/*
 * #####################################################################################
 *                              SIMULATION MODEL
 * #####################################################################################
 *                 LoRaWAN communication             5G sub6GHz communication
 *    © © © © ©                              X X                              ((( ^
 *     © © © © )))       NON3GPP         ((( |O| )))         3GPP                /X\
 *   ©© © © ©                                X X                                /XXX\
 * Devices Lorawan                          UAVs                              BSs 5G/B5G
 * #####################################################################################
 *
 * UAVs aerial bases: Parameter <<startPositions>> of ns-3 simulation
 *  1. UAVs starts in Aerial Bases and moves to the ideal position
 *  2. UAVs starts in optimal positions from optimization algorithm results
 *  3. UAVs starts in the corners of the area
 *  4. UAVs starts in random positions
 * */

#include "ns3/box.h"
#include "ns3/command-line.h"
#include "ns3/core-module.h"
#include "ns3/forwarder-helper.h"
#include "ns3/log.h"
#include "ns3/lora-helper.h"
#include "ns3/lorawan-module.h"
#include "ns3/mobility-helper.h"
#include "ns3/network-server-helper.h"
#include "ns3/node-container.h"
#include "ns3/one-shot-sender-helper.h"
#include "ns3/opengym-module.h"
#include "ns3/propagation-module.h"
#include "ns3/simulator.h"
#include "ns3/stats-module.h"
#include "ns3/traced-value.h"

#include <iomanip>

// QoS, Data rate and Delay
#define MAX_RK 6835.94
#define MIN_RK 183.11

using namespace ns3;
using namespace lorawan;

NS_LOG_COMPONENT_DEFINE("LoRaWAN-OpenAIGym");

Ptr<OpenGymInterface> openGym;
ApplicationContainer applicationContainer = ApplicationContainer();
uint32_t nDevices = 10;
uint32_t nGateways = 1;
uint32_t nPlanes = 1;
uint32_t env_action = 0;
std::vector<std::string> base_actions = {"up", "right", "down", "left", "stay"};
std::vector<std::vector<std::string>> env_action_space;
uint32_t env_action_space_size = 0;
bool env_isGameOver = false;
double areaSide = 0.0;
double movementStep = 0.0;
double envStepTime = 600; // seconds, ns3gym env step time interval
int verbose = 1;
int vcallbacks = 0;
int vtime = 0;
int vresults = 0;
int vgym = 0;
int vmodel = 0;
const int packetSize = 50;
double applicationInterval = 10;
int sumQosNaN = 0;
double simulationStop = 600 * 10 * 50;
int impossible_movement = 0;
uint32_t simSeed = 1;

enum PacketOutcome
{
    _RECEIVED,
    _INTERFERED,
    _NO_MORE_RECEIVERS,
    _UNDER_SENSITIVITY,
    _UNSET
};

struct myPacketStatus
{
    Ptr<const Packet> packet;
    uint32_t senderId;
    uint32_t receiverId;
    Time sentTime;
    Time receivedTime;
    uint8_t senderSF;
    uint8_t receiverSF;
    double senderTP;
    double receiverTP;
    uint32_t outcomeNumber;
    std::vector<enum PacketOutcome> outcomes;
};

std::map<Ptr<const Packet>, myPacketStatus> packetTracker;

NodeContainer endDevices;
NodeContainer gateways;
Ptr<LoraChannel> channel;

// Results computed from trace sources
int pkt_noMoreReceivers = 0;
int pkt_interfered = 0;
int pkt_received = 0;
int pkt_underSensitivity = 0;
int pkt_transmitted = 0;

// Results computed from packet list
int numPackets = 0;
int lostPackets = 0;
int receivedPackets = 0;

/***********************
 * Callback Functions  *
 **********************/
void CheckReceptionByAllGWsComplete(std::map<Ptr<const Packet>, myPacketStatus>::iterator it);
void TransmissionCallback(Ptr<const Packet> packet, uint32_t systemId);
void PacketReceptionCallback(Ptr<const Packet> packet, uint32_t systemId);
void InterferenceCallback(Ptr<const Packet> packet, uint32_t systemId);
void NoMoreReceiversCallback(Ptr<const Packet> packet, uint32_t systemId);
void UnderSensitivityCallback(Ptr<const Packet> packet, uint32_t systemId);
void CourseChangeDetection(std::string context, Ptr<const MobilityModel> model);

/**********************
 * Utility Functions  *
 **********************/
void ScheduleNextStateRead();
void ScheduleNextDataCollect();
void TrackersReset();
Ptr<ListPositionAllocator> NodesPlacement(std::string filename);
void FindNewPositions(std::vector<std::string> env_action_subspace);
Vector FindNewPosition(std::string s_action, Vector uav_position);
uint8_t SFToDR(uint8_t sf);
void setSF_TP(std::string fileConfig);
double GetQoS();
void GenerateActionSpace(std::vector<std::vector<std::string>>& combinations,
                         std::vector<std::string>& currentCombination,
                         int nGat);
std::vector<std::string> GetActionSubspace(uint32_t action, int nGat);

/**********************
 * OPENGYM Functions  *
 **********************/
Ptr<OpenGymSpace> GetObservationSpace();
Ptr<OpenGymSpace> GetActionSpace();
bool ExecuteActions(Ptr<OpenGymDataContainer> action);
Ptr<OpenGymDataContainer> GetObservation();
std::string GetExtraInfo();
float GetReward();
bool GetGameOver();

int
main(int argc, char* argv[])
{
    double reward = 0.0;
    uint32_t openGymPort = 5555;
    bool up = true;
    int startOptimal = 0;
    uint32_t virtualPositions = 1;
    CommandLine cmd;
    cmd.AddValue("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
    cmd.AddValue("nDevices", "Number of end devices to deploy in the simulation", nDevices);
    cmd.AddValue("nGateways", "Number of gateways to deploy in the simulation", nGateways);
    cmd.AddValue("nPlanes",
                 "Number of altitude planes to deploy gateways in the simulation",
                 nPlanes);
    cmd.AddValue("areaSide", "Side of the area to deploy devices", areaSide);
    cmd.AddValue("simSeed", "Random Seed for devices", simSeed);
    cmd.AddValue("reward", "Initial Reward", reward);
    cmd.AddValue("startOptimal", "Start UAVs in optimal positions", startOptimal);
    cmd.AddValue("virtualPositions", "Number of gateway virtual positions", virtualPositions);
    cmd.AddValue("verbose", "Whether to print output or not", verbose);
    cmd.AddValue("vgym", "Whether to print Gym output or not", vgym);
    cmd.AddValue("vresults", "Whether to print results or not", vresults);
    cmd.AddValue("vtime", "Whether to print time control or not", vtime);
    cmd.AddValue("vcallbacks", "Whether to print callbacks or not", vcallbacks);
    cmd.AddValue("vmodel", "Whether to print ns-3 modeling messages or not", vmodel);
    cmd.AddValue("up", "Spread Factor UP", up);

    cmd.Parse(argc, argv);

    movementStep = areaSide / sqrt(virtualPositions);

    /************************************
     *  Logger settings                 *
     ************************************/

    if (verbose)
    {
        LogComponentEnable("LoRaWAN-OpenAIGym", ns3::LOG_LEVEL_ALL);
    }

    RngSeedManager::SetSeed(simSeed);

    Config::SetDefault("ns3::EndDeviceLorawanMac::DRControl", BooleanValue(true));

    /************************
     *  Create the channel  *
     ************************/
    MobilityHelper mobilityED;
    MobilityHelper mobilityGW;
    if (vmodel)
        NS_LOG_INFO("Setting up channel...");
    Ptr<PropagationDelayModel> delay = CreateObject<ConstantSpeedPropagationDelayModel>();
    Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
    loss->SetPathLossExponent(3.76);
    loss->SetReference(1, 10.0);
    channel = CreateObject<LoraChannel>(loss, delay);

    /************************
     *  Create the helpers  *
     ************************/
    if (vmodel)
        NS_LOG_INFO("Setting up helpers...");
    LoraPhyHelper phyHelper = LoraPhyHelper();
    phyHelper.SetChannel(channel);
    LorawanMacHelper macHelper = LorawanMacHelper();
    LoraHelper helper = LoraHelper();
    helper.EnablePacketTracking();

    /************************
     *  Create End Devices  *
     ************************/
    if (vmodel)
        NS_LOG_INFO("Creating end devices...");
    std::string cwd = "/home/rogerio/git/sbrt2024/";
    std::string filename = cwd + "data/results/pl/d_placement_" + std::to_string(simSeed) + "s_" +
                           std::to_string(virtualPositions) + "x" + std::to_string(nPlanes) +
                           "gv_" + std::to_string(nDevices) + "d.dat";
    // Create a set of nodes
    Ptr<ListPositionAllocator> allocatorED = NodesPlacement(filename);
    endDevices.Create(nDevices);
    mobilityED.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobilityED.SetPositionAllocator(allocatorED);
    mobilityED.Install(endDevices);

    // Create the LoraNetDevices of the end devices
    uint8_t nwkId = 54;
    uint32_t nwkAddr = 1864;
    Ptr<LoraDeviceAddressGenerator> addrGen =
        CreateObject<LoraDeviceAddressGenerator>(nwkId, nwkAddr);

    // Create the LoraNetDevices of the end devices
    macHelper.SetAddressGenerator(addrGen);
    phyHelper.SetDeviceType(LoraPhyHelper::ED);
    macHelper.SetDeviceType(LorawanMacHelper::ED_A);
    macHelper.SetRegion(LorawanMacHelper::EU);
    helper.Install(phyHelper, macHelper, endDevices);

    /*********************
     *  Create Gateways  *
     *********************/
    if (vmodel)
        NS_LOG_INFO("Creating gateways...");

    Ptr<ListPositionAllocator> gatewaysPositions = CreateObject<ListPositionAllocator>();

    if (startOptimal == 1)
    {
        // g_placement_10s_4x1gv_50d
        filename = cwd + "data/results/pl/g_placement_" + std::to_string(simSeed) + "s_" +
                   std::to_string(virtualPositions) + "x1gv_" + std::to_string(nDevices) + "d.dat";
        gatewaysPositions = NodesPlacement(filename);
        nGateways = gatewaysPositions->GetSize();
    }
    else
    {
        Ptr<UniformRandomVariable> rd = CreateObject<UniformRandomVariable>();
        rd->SetAttribute("Min", DoubleValue(0.0));
        rd->SetAttribute("Max", DoubleValue(sqrt(virtualPositions) - 1));

        double step = floor(areaSide / ceil(sqrt(virtualPositions)));
        double ini = floor(step / 2);

        for (uint32_t i = 0; i < nGateways; ++i)
        {
            uint32_t rx = rd->GetInteger() * step + ini;
            uint32_t ry = rd->GetInteger() * step + ini;
            gatewaysPositions->Add(Vector(rx, ry, 45));
        }
    }
    gateways.Create(nGateways);
    mobilityGW.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobilityGW.SetPositionAllocator(gatewaysPositions);
    mobilityGW.Install(gateways);

    // Create a net device for each gateway
    phyHelper.SetDeviceType(LoraPhyHelper::GW);
    macHelper.SetDeviceType(LorawanMacHelper::GW);
    helper.Install(phyHelper, macHelper, gateways);

    /**************************************
     * Force ADR or set DR from optimizer  *
     ***************************************/
    // Force ADR
    // ns3::lorawan::LorawanMacHelper::SetSpreadingFactorsUp(endDevices, gateways, channel);
    // Set SF and TP from optimizer
    setSF_TP(cwd + "/data/results/pl/d_configs_" + std::to_string(simSeed) + "s_" +
             std::to_string(virtualPositions) + "x" + std::to_string(nPlanes) + "gv_" +
             std::to_string(nDevices) + "d.dat");

    /***************************
     *  Connect trace sources  *
     ***************************/
    for (auto j = endDevices.Begin(); j != endDevices.End(); ++j)
    {
        Ptr<Node> node = *j;
        Ptr<LoraNetDevice> loraNetDevice = node->GetDevice(0)->GetObject<LoraNetDevice>();
        Ptr<LoraPhy> phy = loraNetDevice->GetPhy();
        phy->TraceConnectWithoutContext("StartSending", MakeCallback(&TransmissionCallback));
    }

    for (auto g = gateways.Begin(); g != gateways.End(); ++g)
    {
        Ptr<Node> object = *g;
        Ptr<NetDevice> netDevice = object->GetDevice(0);
        Ptr<LoraNetDevice> loraNetDevice = netDevice->GetObject<LoraNetDevice>();
        Ptr<GatewayLoraPhy> gwPhy = loraNetDevice->GetPhy()->GetObject<GatewayLoraPhy>();
        // Packets tracing callbacks
        gwPhy->TraceConnectWithoutContext("ReceivedPacket", MakeCallback(&PacketReceptionCallback));
        gwPhy->TraceConnectWithoutContext("LostPacketBecauseInterference",
                                          MakeCallback(&InterferenceCallback));
        gwPhy->TraceConnectWithoutContext("LostPacketBecauseNoMoreReceivers",
                                          MakeCallback(&NoMoreReceiversCallback));
        gwPhy->TraceConnectWithoutContext("LostPacketBecauseUnderSensitivity",
                                          MakeCallback(&UnderSensitivityCallback));
        // Mobility CourseChange Callback
        std::ostringstream oss;
        oss.str("");
        oss << "/NodeList/" << object->GetId() << "/$ns3::MobilityModel/CourseChange";
        Config::Connect(oss.str(), MakeCallback(&CourseChangeDetection));
        if (vcallbacks)
            NS_LOG_INFO("CallBack Connected on: " << oss.str());
    }

    /************************************
     *  Create the openGym environment  *
     ************************************/
    std::vector<std::string> env_action_subspace;
    GenerateActionSpace(env_action_space, env_action_subspace, nGateways);
    env_action_space_size = env_action_space.size();

    if (vgym)
    {
        NS_LOG_INFO("Ns3Env parameters:");
        NS_LOG_INFO("--simulationTime: " << simulationStop);
        NS_LOG_INFO("--openGymPort: " << openGymPort);
        NS_LOG_INFO("--envStepTime: " << envStepTime);
        NS_LOG_INFO("--simulationSeed: " << simSeed);
        NS_LOG_INFO("--envActionSpaceSize: " << env_action_space_size);
    }

    openGym = CreateObject<OpenGymInterface>(openGymPort);
    openGym->SetGetActionSpaceCb(MakeCallback(&GetActionSpace));
    openGym->SetGetObservationSpaceCb(MakeCallback(&GetObservationSpace));
    openGym->SetGetGameOverCb(MakeCallback(&GetGameOver));
    openGym->SetGetObservationCb(MakeCallback(&GetObservation));
    openGym->SetGetRewardCb(MakeCallback(&GetReward));
    openGym->SetGetExtraInfoCb(MakeCallback(&GetExtraInfo));
    openGym->SetExecuteActionsCb(MakeCallback(&ExecuteActions));

    /***************************
     *  Create Network Server  *
     **************************/
    Ptr<Node> networkServer = CreateObject<Node>();
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    P2PGwRegistration_t gwRegistration;
    for (auto gw = gateways.Begin(); gw != gateways.End(); ++gw)
    {
        auto container = p2p.Install(networkServer, *gw);
        auto serverP2PNetDev = DynamicCast<PointToPointNetDevice>(container.Get(0));
        gwRegistration.emplace_back(serverP2PNetDev, *gw);
    }

    NetworkServerHelper nsHelper;
    nsHelper.SetAdr("ns3::AdrComponent");
    nsHelper.EnableAdr(true);
    nsHelper.SetEndDevices(endDevices);
    nsHelper.SetGatewaysP2P(gwRegistration);
    nsHelper.Install(networkServer);

    /*********************************************
     *  Schedule applications                    *
     *********************************************/

    Simulator::Schedule(Seconds(700), &ScheduleNextStateRead);
    ScheduleNextDataCollect();
    if (vmodel)
        NS_LOG_INFO("Completed configuration");
\
    /****************
     *  Simulation  *
     ****************/
    Simulator::Run();
    if (vmodel)
        NS_LOG_INFO("Computing performance metrics...");
    openGym->NotifySimulationEnd();
    Simulator::Destroy();
    if (vmodel)
        NS_LOG_INFO("Simulation finished");
}

uint8_t
SFToDR(uint8_t sf)
{
    return (12 - sf);
}

// Set SF and TP for the sender
void
setSF_TP(std::string fileConfig)
{
    Ptr<LoraPhy> phyED;
    Ptr<ClassAEndDeviceLorawanMac> macED;
    const char* cfg = fileConfig.c_str();
    double id;
    double sf;
    double tp;
    std::ifstream in_File(cfg);
    std::string line{};
    std::getline(in_File, line);
    while (std::getline(in_File, line))
    {
        std::istringstream iss(line);

        std::string substring{};
        std::vector<std::string> substrings{};

        while (std::getline(iss, substring, ','))
        {
            substrings.push_back(substring);
        }

        id = std::stod(substrings[0]);
        sf = std::stod(substrings[1]);
        tp = std::stod(substrings[2]);

        Ptr<Node> node = endDevices.Get(id);
        Ptr<LoraNetDevice> loraNetDevice = node->GetDevice(0)->GetObject<LoraNetDevice>();
        Ptr<LoraPhy> phy = loraNetDevice->GetPhy();
        macED = loraNetDevice->GetMac()->GetObject<ClassAEndDeviceLorawanMac>();
        macED->SetDataRate(SFToDR(sf));
        macED->SetTransmissionPower(tp);
    }
    in_File.close();
}

/**
 * Places the end devices according to the allocator object in the input file..
 * @param filename: output filename
 * @return number of devices
 **/
Ptr<ListPositionAllocator>
NodesPlacement(std::string filename)
{
    double edX = 0.0;
    double edY = 0.0;
    double edZ = 0.0;
    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();
    const char* c = filename.c_str();
    // Get Devices position from File
    std::ifstream in_File(c);
    if (!in_File)
    {
        if (vmodel)
            NS_LOG_INFO("Could not open the file - '" << filename << "'");
    }
    else
    {
        // discard the header line
        std::string line{};
        std::getline(in_File, line);
        while (std::getline(in_File, line))
        {
            std::istringstream iss(line);

            std::string substring{};
            std::vector<std::string> substrings{};

            while (std::getline(iss, substring, ','))
            {
                substrings.push_back(substring);
            }

            edX = std::stod(substrings[1]);
            edY = std::stod(substrings[2]);
            edZ = std::stod(substrings[3]);

            allocator->Add(Vector(edX, edY, edZ));
        }
        in_File.close();
    }
    return allocator;
}

/**
 * Find the new position of the nodes.
 * In case of area violation, a penalty is notified to the GYM
 */

void
FindNewPositions(std::vector<std::string> env_action_subspace)
{
    Vector new_pos;
    Ptr<MobilityModel> gwMob;
    Ptr<ListPositionAllocator> allocator = CreateObject<ListPositionAllocator>();
    impossible_movement = 0;
    for (uint32_t i = 0; i < nGateways && impossible_movement == 0; ++i)
    {
        gwMob = gateways.Get(i)->GetObject<MobilityModel>();
        Vector uav_position = gwMob->GetPosition();
        new_pos = FindNewPosition(env_action_subspace[i], uav_position);
        allocator->Add(new_pos);
    }

    // Check if the new positions cause a collision
    // iterates on allocator to check if there is a collision
    for (uint i = 0; i < allocator->GetSize() - 1 && impossible_movement == 0; ++i)
    {
        Vector pos1 = allocator->GetNext();
        for (uint j = i + 1; j < allocator->GetSize(); ++j)
        {
            Vector pos2 = allocator->GetNext();
            if (pos1.x == pos2.x && pos1.y == pos2.y && pos1.z == pos2.z)
            {
                impossible_movement = 2;
                break;
            }
        }
    }

    // Movement is not allowed in case of collision
    if (impossible_movement == 0) // movement ok - set new positions
    {
        for (uint32_t i = 0; i < nGateways; ++i)
        {
            gwMob = gateways.Get(i)->GetObject<MobilityModel>();
            gwMob->SetPosition(allocator->GetNext());
        }
    }
}

/**
 * Find the new position of the node.
 * In case of area violation, a penalty is notified to the GYM
 */
Vector
FindNewPosition(std::string s_action, Vector uav_position)
{
    // Find a new position from the new state obtained from the GYM
    Vector new_pos = {0.0, 0.0, 0.0};
    Box area_bounds = Box(0.0, areaSide, 0.0, areaSide, 45.0, 45.0);
    // Check the possibility of executing the movement,
    // initially there will be no movements on the z-axis
    // Get the action index from base_actions
    uint32_t action = 4;
    auto it = std::find(base_actions.begin(), base_actions.end(), s_action);
    if (it != base_actions.end())
    { // action found
        uint32_t index = std::distance(base_actions.begin(), it);
        action = index;
    }
    switch (action)
    {
    case 0: // up
        if (uav_position.y + movementStep > area_bounds.yMax)
        {
            impossible_movement = 1;
        }
        else
        {
            new_pos = Vector(uav_position.x, uav_position.y + movementStep, uav_position.z);
        }
        break;
    case 1: // right
        if (uav_position.x + movementStep > area_bounds.xMax)
        {
            impossible_movement = 1;
        }
        else
        {
            new_pos = Vector(uav_position.x + movementStep, uav_position.y, uav_position.z);
        }
        break;
    case 2: // down
        if (uav_position.y - movementStep < area_bounds.yMin)
        {
            impossible_movement = 1;
        }
        else
        {
            new_pos = Vector(uav_position.x, uav_position.y - movementStep, uav_position.z);
        }
        break;

    case 3: // left
        if (uav_position.x - movementStep < area_bounds.xMin)
        {
            impossible_movement = 1;
        }
        else
        {
            new_pos = Vector(uav_position.x - movementStep, uav_position.y, uav_position.z);
        }
        break;
    case 4: // stay
        new_pos = uav_position;
        break;
    }

    return new_pos;
}

double
GetQoS()
{
    // Vector composition:
    // i:: device, j[0] sf j[1] data rate j[2] delay
    std::vector<std::vector<double>> deviceData;
    std::vector<std::vector<std::vector<double>>> deviceSimulatedData;
    double sumQos = 0.0;
    double mean_qos = 0.0;
    if (vresults)
        NS_LOG_INFO("Collecting package data!");

    deviceSimulatedData.reserve(nDevices);
    for (uint32_t i = 0; i < nDevices; ++i)
    {
        deviceSimulatedData.push_back({{}, {}, {}});
    }

    lorawan::LoraTag tag;
    if (vresults)
        NS_LOG_UNCOND("Devices Simulated Results...");

    for (auto p = packetTracker.begin(); p != packetTracker.end(); ++p)
    {
        numPackets += 1;
        (*p).second.packet->PeekPacketTag(tag);
        if ((*p).second.receiverId == 0)
        {
            lostPackets++;
        }
        else
        {
            receivedPackets++;
        }

        double size = (*p).second.packet->GetSize() * 8; // bits
        int devID = (*p).second.senderId;
        double sf;
        sf = ((*p).second.senderSF);
        double dk;
        dk = ((*p).second.receivedTime.GetSeconds() - (*p).second.sentTime.GetSeconds());
        double rk;
        rk = (size / ((*p).second.receivedTime.GetSeconds() - (*p).second.sentTime.GetSeconds()));
        deviceSimulatedData[devID][0].push_back(sf);
        deviceSimulatedData[devID][1].push_back(rk);
        deviceSimulatedData[devID][2].push_back(dk);
    }
    if (numPackets == 0)
    {
        return -1;
    }
    std::vector<std::vector<double>> deviceSummarizedData;
    sumQos = 0.0;
    int qosNaN = 0;

    for (uint32_t devID = 0; devID < nDevices; ++devID)
    {
        if (deviceSimulatedData[devID][0].empty())
        {
            continue;
        }
        double dk = 0;
        double rk = 0;
        double sf = deviceSimulatedData[devID][0][0];
        double qos = 0;
        int qtd = 0;
        // depurar aqui prá contornar o SIGSEGV
        // O VETOR NÃO TÁ COMPLETO
        for (unsigned int i = 0; i < deviceSimulatedData[devID][0].size();)
        {
            if (deviceSimulatedData[devID][2].at(i) > 0.0)
            {
                rk += deviceSimulatedData[devID][1].at(i);
                dk += deviceSimulatedData[devID][2].at(i);
                qtd += 1;
                i++;
            }
            else
            {
                deviceSimulatedData[devID][0].erase(deviceSimulatedData[devID][0].begin() + i);
                deviceSimulatedData[devID][1].erase(deviceSimulatedData[devID][1].begin() + i);
                deviceSimulatedData[devID][2].erase(deviceSimulatedData[devID][2].begin() + i);
            }
        }
        // Device QoS
        qos = (rk / qtd) / MAX_RK + (1 - ((dk / qtd) / MIN_RK));
        deviceSummarizedData.push_back({sf, rk / qtd, dk / qtd, qos});
        if (isNaN(qos))
        {
            if (vresults)
                NS_LOG_UNCOND("Device " << devID << " does not meet QoS criteria!");
            qosNaN++;
        }
        else
        {
            sumQos += qos;
        }
    }
    if (vresults)
        NS_LOG_UNCOND(qosNaN << " devices does not meet QoS criteria!");
    sumQosNaN += qosNaN;
    // Gateways QoS
    mean_qos = sumQos / nDevices;

    if (deviceSummarizedData.size() < nDevices)
    {
        if (vresults)
            NS_LOG_UNCOND("There are unserved devices!");
        return mean_qos;
    }
    else
    {
        for (uint32_t i = 0; i < nDevices; ++i)
        {
            if (vresults)
                NS_LOG_UNCOND(i << " " << deviceSummarizedData[i][0] << " "
                                << deviceSummarizedData[i][1] << " " << deviceSummarizedData[i][2]
                                << " " << deviceSummarizedData[i][3]);
        }
    }
    if (mean_qos < 0) // || mean_qos < QOS_THRESHOLD)
    {
        if (vresults)
            NS_LOG_UNCOND("Simulation instance does not meet QoS criteria!");
    }
    else
    {
        if (vresults)
            NS_LOG_UNCOND("Simulated QoS: " << mean_qos);
    }
    if (vresults)
        NS_LOG_UNCOND("Summary: " << numPackets << " Sent: " << pkt_transmitted
                                  << " Lost: " << lostPackets << " Received: " << receivedPackets
                                  << " QoS: " << (isNaN(mean_qos) ? -1 : mean_qos));
    if (vresults)
        NS_LOG_UNCOND("Callback results: Tx " << pkt_transmitted << " Rx " << pkt_received
                                              << " UnderSensitivity " << pkt_underSensitivity
                                              << " NoMoreReceivers " << pkt_noMoreReceivers
                                              << " Interfered " << pkt_interfered);

    return mean_qos;
}

void
CheckReceptionByAllGWsComplete(std::map<Ptr<const Packet>, myPacketStatus>::iterator it)
{
    // Check whether this packet is received by all gateways
    if ((*it).second.outcomeNumber == nGateways)
    {
        // Update the statistics
        myPacketStatus status = (*it).second;
        for (uint32_t j = 0; j < nGateways; j++)
        {
            switch (status.outcomes.at(j))
            {
            case _RECEIVED: {
                pkt_received += 1;
                break;
            }
            case _UNDER_SENSITIVITY: {
                pkt_underSensitivity += 1;
                break;
            }
            case _NO_MORE_RECEIVERS: {
                pkt_noMoreReceivers += 1;
                break;
            }
            case _INTERFERED: {
                pkt_interfered += 1;
                break;
            }
            case _UNSET: {
                break;
            }
            }
        }
        // Remove the packet from the tracker
        //              packetTracker.erase (it);
    }
}

void
TransmissionCallback(Ptr<const Packet> packet, uint32_t systemId)
{
    if (vcallbacks)
        NS_LOG_INFO("Transmitted a packet from device " << systemId << " at "
                                                        << Simulator::Now().GetSeconds());
    LoraTag tag;
    packet->PeekPacketTag(tag);

    myPacketStatus status;
    status.packet = packet;
    status.senderId = systemId;
    status.sentTime = Simulator::Now();
    status.outcomeNumber = 0;
    status.outcomes = std::vector<enum PacketOutcome>(nGateways, _UNSET);
    status.senderSF = tag.GetSpreadingFactor();

    packetTracker.insert(std::pair<Ptr<const Packet>, myPacketStatus>(packet, status));
    pkt_transmitted += 1;
}

void
PacketReceptionCallback(Ptr<const Packet> packet, uint32_t systemId)
{
    // Remove the successfully received packet from the list of sent ones
    if (vcallbacks)
        NS_LOG_INFO("A packet was successfully received at gateway "
                    << systemId << " at " << Simulator::Now().GetSeconds());
    LoraTag tag;
    packet->PeekPacketTag(tag);

    auto it = packetTracker.find(packet);

    if (it != packetTracker.end())
    {
        if ((*it).first == packet)
        {
            (*it).second.outcomes.at(systemId - nDevices) = _RECEIVED;
            (*it).second.outcomeNumber += 1;
            if ((*it).second.receivedTime == Seconds(0))
            {
                (*it).second.receivedTime = Simulator::Now();
                (*it).second.receiverId = systemId;
                //          cc++;
            }
            (*it).second.receiverSF = tag.GetSpreadingFactor();
            (*it).second.receiverTP = tag.GetReceivePower();

            CheckReceptionByAllGWsComplete(it);
        }
    }
}

void
InterferenceCallback(Ptr<const Packet> packet, uint32_t systemId)
{
    if (vcallbacks)
        NS_LOG_INFO("A packet was lost because of interference at gateway "
                    << systemId << " at " << Simulator::Now().GetSeconds());
    auto it = packetTracker.find(packet);
    if ((*it).first == packet)
    {
        (*it).second.outcomes.at(systemId - nDevices) = _INTERFERED;
        (*it).second.outcomeNumber += 1;
        CheckReceptionByAllGWsComplete(it);
    }
}

void
NoMoreReceiversCallback(Ptr<const Packet> packet, uint32_t systemId)
{
    if (vcallbacks)
        NS_LOG_INFO("A packet was lost because there were no more receivers at gateway "
                    << systemId << " at " << Simulator::Now().GetSeconds());
    auto it = packetTracker.find(packet);

    if ((*it).first == packet)
    {
        (*it).second.outcomes.at(systemId - nDevices) = _NO_MORE_RECEIVERS;
        (*it).second.outcomeNumber += 1;
    }
    CheckReceptionByAllGWsComplete(it);
}

void
UnderSensitivityCallback(Ptr<const Packet> packet, uint32_t systemId)
{
    if (vcallbacks)
        NS_LOG_INFO("A packet arrived at the gateway " << systemId << " under sensitivity."
                                                       << " at " << Simulator::Now().GetSeconds());
    auto it = packetTracker.find(packet);
    if ((*it).first == packet)
    {
        (*it).second.outcomes.at(systemId - nDevices) = _UNDER_SENSITIVITY;
        (*it).second.outcomeNumber += 1;
        CheckReceptionByAllGWsComplete(it);
    }
}

void
CourseChangeDetection(std::string context, Ptr<const MobilityModel> model)
{
    Vector uav_position = model->GetPosition();
    if (vcallbacks)
        NS_LOG_INFO(context << " x = " << uav_position.x << ", y = " << uav_position.y
                            << ", z = " << uav_position.z);
}

/**
 * OPENGYM ENVIRONMENT FUNCTIONS
 * */

void
ScheduleNextStateRead()
{
    if (vtime)
        NS_LOG_INFO("NowNSR: " << Simulator::Now().GetSeconds());
    Simulator::Schedule(Seconds(700), &ScheduleNextStateRead);
    openGym->NotifyCurrentState();
}

void
ScheduleNextDataCollect()
{
    if (vtime)
        NS_LOG_INFO("NowNDC: " << Simulator::Now().GetSeconds());
    TrackersReset();
    PeriodicSenderHelper periodicSenderHelper;
    periodicSenderHelper.SetPeriod(Seconds(applicationInterval));
    periodicSenderHelper.SetPacketSize(packetSize);
    ForwarderHelper forHelper = ForwarderHelper();
    forHelper.Install(gateways);
    applicationContainer = periodicSenderHelper.Install(endDevices);
    applicationContainer.Start(Seconds(0));
    applicationContainer.Stop(Seconds(600));
    // Force ADR
    ns3::lorawan::LorawanMacHelper::SetSpreadingFactorsUp(endDevices, gateways, channel);
}

void
TrackersReset()
{
    packetTracker.clear();
    pkt_transmitted = 0;
    pkt_received = 0;
    pkt_interfered = 0;
    pkt_noMoreReceivers = 0;
    pkt_underSensitivity = 0;
    numPackets = 0;
    lostPackets = 0;
    receivedPackets = 0;
    //    if (vtime) NS_LOG_INFO("Trackers Reseted!");
}

Ptr<OpenGymSpace>
GetActionSpace()
{
    /**
     * The action space contains four expected actions: move up,
     * move down, move left, and move right and the UAV number.
     * The UAV number is used to identify the UAV that will execute the action.
     * Action number represents the uav number * 4 + the action.
     * Ex: 11 = 2 * 4 + 3 = UAV 2 move right
     * The action space is represented by the following enum:
     * enum ActionMovements{
     *                      MOVE_UP,
     *                      MOVE_DOWN,
     *                      MOVE_RIGHT,
     *                      MOVE_LEFT,
     *                      KEPT_STOPPED
     * }
     * **/
    if (vgym)
        NS_LOG_FUNCTION("GetActionSpace");
    Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace>(env_action_space_size);
    if (vgym)
        NS_LOG_INFO("GetActionSpace: " << space << " Time: " << Simulator::Now().GetSeconds());
    return space;
}

bool
GetGameOver()
{
    if (vgym)
        NS_LOG_INFO("MyGetGameOver: " << env_isGameOver);
    if (impossible_movement == 2)
    {
        env_isGameOver = true;
    }
    return env_isGameOver;
}

float
GetReward()
{
    double m_qos = 0.0;
    sumQosNaN = 0;
    if (env_action < env_action_space_size)
    {
        m_qos = GetQoS();
        m_qos = (impossible_movement == 2)   ? -1                      // Collision detected
                : (impossible_movement == 1) ? 0                       // Movement not allowed
                                             : (isNaN(m_qos) ? 0    // QoS not met
                                                             : m_qos);// QoS ok
    }
    if (vgym)
        NS_LOG_INFO("MyGetReward: " << m_qos);
    return m_qos;
}

std::string
GetExtraInfo()
{
    std::string env_info = "";
    env_info = "[T.Pack. " + std::to_string(numPackets) + ", Rec.Pack. " +
               std::to_string(receivedPackets) + ", Lost.Pack. " + std::to_string(lostPackets) +
               " seed: " + std::to_string(simSeed) + "]";
    if (impossible_movement == 1)
    {
        env_info += "[impossible movement]";
    }
    else if (impossible_movement == 2)
    {
        env_info += "[Collision detected]";
    }
    if (sumQosNaN > 0)
    {
        env_info = env_info + "[Unnat. Dev. " + std::to_string(sumQosNaN) + "]";
    }
    if (vgym)
        NS_LOG_INFO("MyGetExtraInfo: " << env_info);
    return env_info;
}

bool
ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
    env_action = discrete->GetValue();
    // Identify UAVs movements from action
    // Get action subspace from action space e.g. ("Up", "Down", ...) for each drone
    std::vector<std::string> env_action_subspace = GetActionSubspace(env_action, nGateways);
    FindNewPositions(env_action_subspace);
    ScheduleNextDataCollect();
    if (vgym)
    {
        std::string info = "MyExecuteAction: [";
        for (const auto& a : env_action_subspace)
        {
            info += a + ", ";
        }
        info += "]";
        NS_LOG_INFO(std::endl << info);
    }
    return true;
}

Ptr<OpenGymDataContainer>
GetObservation()
{
    uint32_t parameterNum = nGateways;
    std::vector<uint32_t> shape = {
        parameterNum,
    };

    Ptr<OpenGymBoxContainer<uint32_t>> box = CreateObject<OpenGymBoxContainer<uint32_t>>(shape);
    for (auto g = gateways.Begin(); g != gateways.End(); ++g)
    {
        Ptr<Node> object = *g;
        Ptr<NetDevice> netDevice = object->GetDevice(0);
        Ptr<MobilityModel> mobility = netDevice->GetNode()->GetObject<MobilityModel>();
        Vector uav_position = mobility->GetPosition();
        box->AddValue(uav_position.x);
        box->AddValue(uav_position.y);
        box->AddValue(uav_position.z);
    }
    if (vgym)
        NS_LOG_INFO("MyGetObservation: " << box);
    return box;
}

Ptr<OpenGymSpace>
GetObservationSpace()
{
    uint32_t parameterNum = nGateways;
    std::vector<uint32_t> shape = {
        parameterNum,
    };
    float low = 0.0;
    float high = 10000.0;
    std::string dtype = TypeNameGet<uint32_t>();
    Ptr<OpenGymBoxSpace> box = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
    if (vgym)
        NS_LOG_INFO("MyGetObservationSpace: " << box);
    return box;
}

void
GenerateActionSpace(std::vector<std::vector<std::string>>& combinations,
                    std::vector<std::string>& currentCombination,
                    int nGat)
{
    if (nGat == 0)
    {
        combinations.push_back(currentCombination);
        return;
    }

    for (const auto& movement : base_actions)
    {
        currentCombination.push_back(movement);
        GenerateActionSpace(combinations, currentCombination, nGat - 1);
        currentCombination.pop_back();
    }
}

std::vector<std::string>
GetActionSubspace(uint32_t action, int nGat)
{
    std::vector<std::string> env_action_subspace;
    env_action_subspace.reserve(nGat);
    for (int i = 0; i < nGat; i++)
    {
        env_action_subspace.push_back(env_action_space[action][i]);
    }
    return env_action_subspace;
}
