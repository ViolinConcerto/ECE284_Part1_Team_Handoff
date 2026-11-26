module core #(
    parameter bw = 4,
    parameter psum_bw = 32,
    parameter col = 8,
    parameter row = 8
)(
    input clk, reset, compute_clear,
    input [33:0] inst,
    input [row*bw-1:0] D_xmem,
    input act_signed,
    input sfp_clear, sfp_relu, pmem_from_sfp,
    input [4:0] sfp_shift,
    output ofifo_valid, l0_ready, ofifo_full,
    output [col*psum_bw-1:0] sfp_out, pmem_read_data,
    output reg [31:0] weight_load_cycles,
    output reg [31:0] weight_load_sessions
);
    wire [row*bw-1:0] l0_in;
    wire [col*psum_bw-1:0] sfp_in, ofifo_out, pmem_write_data;
    wire l0_full, ofifo_ready;
    reg previous_load;
    assign pmem_read_data = sfp_in;
    corelet #(.row(row), .col(col), .bw(bw), .psum_bw(psum_bw)) corelet_instance(
        .clk(clk), .reset(reset), .compute_clear(compute_clear), .inst(inst),
        .act_signed(act_signed), .sfp_clear(sfp_clear), .sfp_relu(sfp_relu),
        .sfp_shift(sfp_shift), .pmem_from_sfp(pmem_from_sfp),
        .l0_in(l0_in), .sfp_in(sfp_in), .l0_o_full(l0_full), .l0_o_ready(l0_ready),
        .ofifo_o_full(ofifo_full), .ofifo_o_ready(ofifo_ready),
        .ofifo_valid(ofifo_valid), .ofifo_out(ofifo_out), .sfp_out(sfp_out),
        .pmem_write_data(pmem_write_data));
    video_sram #(.data_bw(row*bw)) input_sram(
        .CLK(clk), .D(D_xmem), .Q(l0_in), .CEN(inst[19]), .WEN(inst[18]), .A(inst[17:7]));
    video_sram #(.data_bw(col*psum_bw)) psum_sram(
        .CLK(clk), .D(pmem_write_data), .Q(sfp_in),
        .CEN(inst[32]), .WEN(inst[31]), .A(inst[30:20]));
    always @(posedge clk) begin
        if (reset) begin
            previous_load <= 0;
            weight_load_cycles <= 0;
            weight_load_sessions <= 0;
        end else begin
            previous_load <= inst[0];
            if (inst[0] && !compute_clear) begin
                weight_load_cycles <= weight_load_cycles + 1;
                if (!previous_load) weight_load_sessions <= weight_load_sessions + 1;
            end
        end
    end
endmodule
