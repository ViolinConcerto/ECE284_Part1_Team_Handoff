module core (clk, reset, inst, D_xmem, ofifo_valid, sfp_out,
             sfp_clear, sfp_relu, pmem_from_sfp, pmem_read_data,
             l0_ready, ofifo_full);
    parameter bw = 4;
    parameter psum_bw = 16;
    parameter col = 8;
    parameter row = 8;

    //input and output signals based on given testbench    
    input clk;
    input reset;
    input [33:0] inst;
    input [31:0] D_xmem;
    input sfp_clear;
    input sfp_relu;
    input pmem_from_sfp;
    output ofifo_valid;
    output [127:0] sfp_out;
    output [127:0] pmem_read_data;
    output l0_ready;
    output ofifo_full;
    
    //wire definition
    wire [127:0] sfp_in;
    wire [31:0] l0_in;
    wire l0_o_full;
    wire l0_o_ready;
    wire [127:0] ofifo_out;
    wire ofifo_o_full;
    wire ofifo_o_ready;
    wire CEN_xmem;
    wire WEN_xmem;
    wire CEN_pmem;
    wire WEN_pmem;
    wire [10:0] A_pmem;
    wire [10:0] A_xmem;
    wire [127:0] pmem_write_data;
    assign pmem_read_data = sfp_in;
    assign l0_ready = l0_o_ready;
    assign ofifo_full = ofifo_o_full;

    //inst[0:3],6,33 already declared in corelet.v;inst[5:4] not used for weight stationary
    assign CEN_pmem = inst[32];
    assign WEN_pmem = inst[31];
    assign A_pmem = inst[30:20];
    assign CEN_xmem = inst[19];
    assign WEN_xmem = inst[18];
    assign A_xmem = inst[17:7];

    //Instantiation
    corelet #(.row(row), .col(col), .bw(bw), .psum_bw(psum_bw)) corelet_instance(
        .clk(clk),
        .reset(reset),
        .ofifo_valid(ofifo_valid),
        .sfp_in(sfp_in),
        .sfp_out(sfp_out),
        .inst(inst),
        .l0_in(l0_in),
        .l0_o_full(l0_o_full),
        .l0_o_ready(l0_o_ready),
        .ofifo_out(ofifo_out),
        .ofifo_o_full(ofifo_o_full),
        .ofifo_o_ready(ofifo_o_ready),
        .sfp_clear(sfp_clear),
        .sfp_relu(sfp_relu),
        .pmem_from_sfp(pmem_from_sfp),
        .pmem_write_data(pmem_write_data));
    
    sram_32b_w2048 input_sram(
        .CLK(clk), 
        .D(D_xmem), 
        .Q(l0_in), 
        .CEN(CEN_xmem), 
        .WEN(WEN_xmem), 
        .A(A_xmem));

    sram_128b_w2048 psum_sram(
        .CLK(clk), 
        .D(pmem_write_data),
        .Q(sfp_in), 
        .CEN(CEN_pmem), 
        .WEN(WEN_pmem), 
        .A(A_pmem));


endmodule
