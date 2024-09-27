
#include <iostream>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

int main() {
    int clientSocket;
    sockaddr_in serverAddr;
    const std::string serverIp = "192.168.42.63"; 
    const int serverPort = 8080;

    // Create a socket
    clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket < 0) {
        std::cerr << "Failed to create socket\n";
        return -1;
    }

    // Define server address
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(serverPort);
    if (inet_pton(AF_INET, serverIp.c_str(), &serverAddr.sin_addr) <= 0) {
        std::cerr << "Invalid address or Address not supported\n";
        close(clientSocket);
        return -1;
    }

    // Connect to the server
    if (connect(clientSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Connection failed\n";
        close(clientSocket);
        return -1;
    }

    while (true) {
        // Send message
        std::string message;
        std::cout << "Enter a message: ";
        std::getline(std::cin, message);
        send(clientSocket, message.c_str(), message.size(), 0);

        // Receive response
        char buffer[4096] = {0};
        int bytesReceived = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0'; // Null-terminate the received data
            std::cout << "Server response: " << buffer << std::endl;
        } else {
            std::cerr << "Failed to receive data\n";
            break;
        }
    }

    // Close the socket
    close(clientSocket);
    return 0;
}
  