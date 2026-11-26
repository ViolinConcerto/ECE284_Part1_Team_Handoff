module video_sram #(
    parameter data_bw = 32,
    parameter address_bw = 11,
    parameter num = 2048
)(
    input CLK, CEN, WEN,
    input [data_bw-1:0] D,
    input [address_bw-1:0] A,
    output reg [data_bw-1:0] Q
);
    reg [data_bw-1:0] memory [0:num-1];
    always @(posedge CLK) begin
        if (!CEN && WEN) Q <= memory[A];
        if (!CEN && !WEN) memory[A] <= D;
    end
endmodule
